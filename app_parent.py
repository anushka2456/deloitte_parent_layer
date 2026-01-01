import os
import joblib
import requests
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from google import genai

# ---------------- CONFIG ----------------

MODEL_URL = os.getenv("PARENT_MODEL_URL")  # hosted joblib
PREPROC_URL = os.getenv("PARENT_PREPROC_URL")

MODEL_PATH = "parent_model.joblib"
PREPROC_PATH = "parent_preprocessor.joblib"

CAREERS_CSV = "careers.csv"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------------- APP ----------------

app = FastAPI(title="Parent Layer Scoring API")

# ---------------- LOADERS ----------------

def download_if_missing(url: str, path: str):
    if not os.path.exists(path):
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)

def safe_load_models():
    if not MODEL_URL or not PREPROC_URL:
        raise RuntimeError("Model URLs not set in environment")

    download_if_missing(MODEL_URL, MODEL_PATH)
    download_if_missing(PREPROC_URL, PREPROC_PATH)

    model = joblib.load(MODEL_PATH)
    preproc = joblib.load(PREPROC_PATH)
    return model, preproc

model, preprocessor = safe_load_models()
careers_df = pd.read_csv(CAREERS_CSV)

# ---------------- GEMINI ----------------

def explain_with_gemini(career_id: str, score: float) -> str:
    if not GEMINI_API_KEY:
        return "Explanation unavailable."

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = (
            f"Explain briefly why parents would prefer the career '{career_id}'. "
            f"The compatibility score is {score:.2f}. Focus on finances, stability, prestige."
        )
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt
        )
        return response.text.strip()
    except Exception:
        return "Explanation unavailable."

# ---------------- SCHEMA ----------------

class ParentInput(BaseModel):
    budget_max_tuition: float
    importance_finances: int
    importance_job_security: int
    importance_prestige: int
    parent_risk_tolerance: int
    influence_from_people: int
    migration_allowed: bool
    location_preference: str
    unacceptable_careers: List[str]

# ---------------- ENDPOINT ----------------

@app.post("/rescore-parent")
def rescore_parent(data: ParentInput):
    df = careers_df.copy()

    df["f_is_unacceptable"] = df["career_id"].isin(data.unacceptable_careers).astype(int)
    df["f_tuition_affordable"] = (df["tuition"] <= data.budget_max_tuition).astype(int)

    for col, val in {
        "importance_finances": data.importance_finances,
        "importance_job_security": data.importance_job_security,
        "importance_prestige": data.importance_prestige,
        "parent_risk_tolerance": data.parent_risk_tolerance,
        "influence_from_people": data.influence_from_people,
    }.items():
        df[col] = val / 5.0

    X = preprocessor.transform(df)
    scores = model.predict(X)
    df["parent_score"] = scores

    top5 = (
        df.sort_values("parent_score", ascending=False)
        .head(5)[["career_id", "parent_score"]]
        .to_dict(orient="records")
    )

    best = top5[0]
    explanation = explain_with_gemini(best["career_id"], best["parent_score"])

    return {
        "top_5_parent_scores": top5,
        "final_recommendation": {
            "career_id": best["career_id"],
            "parent_score": round(float(best["parent_score"]), 3),
            "parent_explanation": explanation,
        },
    }
