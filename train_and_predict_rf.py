#!/usr/bin/env python3
"""
train_and_predict_rf.py

Usage:
  # Train:
  python train_and_predict_rf.py --train --data parent_only_synthetic_dataset.csv --out models/parent_layer --preproc models/preprocessors

  # Predict (JSON inline):
  python train_and_predict_rf.py --predict --model models/parent_layer/parent_model_rf.joblib --preproc models/preprocessors/parent_preprocessor_coltransformer.joblib \
    --input '{"f_fin_ratio":1.2,"f_risk_penalty":1.0,"p_financial_weight":0.75,"p_job_security_weight":1.0,"p_prestige_weight":0.5,"p_parent_risk_tolerance":0.25,"p_weight_on_parent_layer":0.2,"p_budget_max_tuition":150000,"c_avg_salary":70000,"c_job_security":0.8,"c_prestige":0.7,"c_tuition":20000,"f_location_match":1,"f_tuition_affordable":1,"f_migration_ok":1,"f_is_unacceptable":0}'

  # Predict (interactive):
  python train_and_predict_rf.py --predict --model models/parent_layer/parent_model_rf.joblib --preproc models/preprocessors/parent_preprocessor_coltransformer.joblib
"""

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

FEATURE_COLS = [
    "f_fin_ratio","f_risk_penalty",
    "p_financial_weight","p_job_security_weight","p_prestige_weight",
    "p_parent_risk_tolerance","p_weight_on_parent_layer","p_budget_max_tuition",
    "c_avg_salary","c_job_security","c_prestige","c_tuition",
    "f_location_match","f_tuition_affordable","f_migration_ok","f_is_unacceptable"
]
TARGET_COL = "parent_score"

def build_preprocessor(feature_cols):
    """Return a ColumnTransformer that imputes numeric features and scales to [0,1]."""
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ])
    preproc = ColumnTransformer([
        ("num", num_transformer, feature_cols)
    ], remainder="drop")
    return preproc

def train(data_path: Path, out_model_dir: Path, preproc_dir: Path, n_trees=200, random_state=42):
    df = pd.read_csv(data_path)
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(float).copy()

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=random_state)

    preproc = build_preprocessor(FEATURE_COLS)
    X_train_p = preproc.fit_transform(X_train)
    X_test_p = preproc.transform(X_test)

    # Train RandomForest
    rf = RandomForestRegressor(n_estimators=n_trees, random_state=random_state, n_jobs=-1)
    rf.fit(X_train_p, y_train)

    # Evaluate
    y_pred = rf.predict(X_test_p)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    print("Training complete. Metrics:", metrics)

    # Persist artifacts
    out_model_dir.mkdir(parents=True, exist_ok=True)
    preproc_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_model_dir / "parent_model_rf.joblib"
    preproc_path = preproc_dir / "parent_preprocessor_coltransformer.joblib"

    joblib.dump(rf, model_path)
    joblib.dump(preproc, preproc_path)

    print("Saved model to:", model_path)
    print("Saved preprocessor to:", preproc_path)

    return {"model_path": str(model_path), "preproc_path": str(preproc_path), "metrics": metrics}

def _parse_input_json(json_str: str) -> dict:
    try:
        data = json.loads(json_str)
    except Exception as e:
        raise ValueError(f"Invalid JSON input: {e}")
    # ensure all features present; if missing keep NaN to let imputer handle
    row = {c: data.get(c, np.nan) for c in FEATURE_COLS}
    return row

def interactive_prompt() -> dict:
    print("Interactive input mode. Enter values for features when prompted.")
    row = {}
    for f in FEATURE_COLS:
        while True:
            raw = input(f"{f} (press Enter to set missing): ").strip()
            if raw == "":
                row[f] = np.nan
                break
            try:
                # allow boolean-like 0/1
                if raw.lower() in ("0","1"):
                    row[f] = float(raw)
                else:
                    row[f] = float(raw)
                break
            except ValueError:
                print("Invalid value; enter numeric (e.g., 0.75, 1, 150000) or leave blank for missing.")
    return row

def predict_single(model_path: Path, preproc_path: Path, input_row: dict):
    model = joblib.load(model_path)
    preproc = joblib.load(preproc_path)

    df = pd.DataFrame([input_row], columns=FEATURE_COLS)
    X_proc = preproc.transform(df)
    pred = model.predict(X_proc)[0]
    pred = float(np.clip(pred, 0.0, 1.0))

    # simple explanation: check top contributing engineered cues
    explanations = []
    if input_row.get("f_is_unacceptable", 0) == 1:
        explanations.append("Vetoed by parent.")
    else:
        if input_row.get("f_fin_ratio", 0) >= 1:
            explanations.append("Meets financial preference")
        if input_row.get("c_job_security", 0) >= 0.8:
            explanations.append("High job security")
        if input_row.get("c_prestige", 0) >= 0.75:
            explanations.append("High prestige")
        if input_row.get("f_location_match", 0) == 1:
            explanations.append("Location matches")
    explanation = "; ".join(explanations) if explanations else "Moderate fit"

    return {"prediction": pred, "explanation": explanation}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train a RandomForest model on the CSV.")
    parser.add_argument("--predict", action="store_true", help="Load model+preprocessor and predict from input.")
    parser.add_argument("--data", type=str, default="parent_only_synthetic_dataset.csv", help="CSV dataset path")
    parser.add_argument("--out", type=str, default="models/parent_layer", help="Where to save model")
    parser.add_argument("--preproc", type=str, default="models/preprocessors", help="Where to save preprocessor")
    parser.add_argument("--model", type=str, default="models/parent_layer/parent_model_rf.joblib", help="Model path (for predict)")
    parser.add_argument("--preproc-file", dest="preproc_file", type=str, default="models/preprocessors/parent_preprocessor_coltransformer.joblib", help="Preprocessor path (for predict)")
    parser.add_argument("--input", type=str, help="Input JSON string for prediction (single example). If omitted, interactive prompts will run.")
    parser.add_argument("--trees", type=int, default=200, help="Number of trees for RandomForest")
    args = parser.parse_args()

    if args.train:
        data_path = Path(args.data)
        out_dir = Path(args.out)
        preproc_dir = Path(args.preproc)
        print("Training with data:", data_path)
        res = train(data_path, out_dir, preproc_dir, n_trees=args.trees)
        print("Saved:", res)
        return

    if args.predict:
        model_path = Path(args.model)
        preproc_path = Path(args.preproc_file)
        if not model_path.exists():
            raise SystemExit(f"Model not found at: {model_path}")
        if not preproc_path.exists():
            raise SystemExit(f"Preprocessor not found at: {preproc_path}")

        if args.input:
            input_row = _parse_input_json(args.input)
        else:
            input_row = interactive_prompt()

        out = predict_single(model_path, preproc_path, input_row)
        print("Prediction:", out["prediction"])
        print("Explanation:", out["explanation"])
        return

    parser.print_help()

if __name__ == "__main__":
    main()
