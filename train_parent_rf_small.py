import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# -------- PATHS --------
DATA_PATH = "parent_only_synthetic_dataset.csv"
PREPROCESSOR_PATH = "models/preprocessors/parent_preprocessor.joblib"
MODEL_OUT = "models/parent_layer/parent_model_rf.joblib"

# -------- LOAD DATA --------
df = pd.read_csv(DATA_PATH)

# CHANGE THIS if your label column name is different
TARGET_COL = "career_id"

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# -------- LOAD PREPROCESSOR --------
print("Loading preprocessor...")
preprocessor = joblib.load(PREPROCESSOR_PATH)

X_proc = preprocessor.transform(X)

# -------- TRAIN SMALL RF --------
print("Training small RandomForest...")
model = RandomForestClassifier(
    n_estimators=30,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_proc, y)

# -------- SAVE MODEL --------
Path("models/parent_layer").mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_OUT)

print("✅ Small model trained and saved successfully")
