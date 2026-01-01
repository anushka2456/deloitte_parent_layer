import os
import json
import joblib
import requests
from fastapi import FastAPI, HTTPException

# ========== CONFIG ==========
MODEL_PATH = "models/parent_layer/parent_model_rf.joblib"
PREPROC_PATH = "models/preprocessors/parent_preprocessor.joblib"

# Optional: Google Drive direct download links
MODEL_URL = os.getenv("MODEL_URL")      # set in Render env vars
PREPROC_URL = os.getenv("PREPROC_URL")  # set in Render env vars
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ✅ NEW Gemini SDK (NOT deprecated)
from google import genai

# -------------------------
# CONFIG
# -------------------------

MODEL_DIR = "models/runtime"
MODEL_PATH = os.path.join(MODEL_DIR, "parent_model.joblib")
PREPROC_PATH = os.path.join(MODEL_DIR, "parent_preprocessor.joblib")
CAREERS_CSV = "data/careers.csv"  # small metadata file, safe in git

# 🔴 REPLACE THESE WITH YOUR OWN DRIVE LINKS
MODEL_URL = "https://drive.google.com/file/d/1ZnKTrw9LoJUnx4Bu4URxNEcanJ8PZdYF/view?usp=drive_link"
PREPROC_URL = "https://drive.google.com/file/d/19LE1Q9fJl4dVjdlSkePu87hUmR_IWDar/view?usp=drive_link"

GEMINI_MODEL = "gemini-2.5-pro"

# -------------------------
# APP INIT
# -------------------------

app = FastAPI(title="Parent Layer API")

model = None
preprocessor = None
careers_df = None
llm = None

USE_GEMINI = bool(os.getenv("GEMINI_API_KEY"))

# ========== GEMINI (SAFE) ==========
def explain_with_gemini(career_id: str, score: float):
    if not USE_GEMINI:
        return "Explanation unavailable (Gemini disabled)."

    try:
        from google import genai

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        prompt = (
            f"Explain why career '{career_id}' is suitable "
            f"based on a parent score of {score:.2f}."
        )

        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
        )

        return response.text.strip()

    except Exception as e:
        return f"Explanation unavailable (Gemini error)."

# ========== HELPERS ==========
def download_if_missing(path: str, url: str):
    if os.path.exists(path):
        return

    if not url:
        raise RuntimeError(f"Missing file {path} and no download URL provided.")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    print(f"Downloading {path}...")
    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

def load_assets():
    download_if_missing(MODEL_PATH, MODEL_URL)
    download_if_missing(PREPROC_PATH, PREPROC_URL)

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROC_PATH)
    return model, preprocessor

# ========== APP ==========
app = FastAPI()

model, preprocessor = load_assets()

@app.post("/score")
def score_parent(payload: dict):
    try:
        X = preprocessor.transform([payload])
        score = float(model.predict_proba(X)[0][1])

        explanation = explain_with_gemini(
            payload.get("career_id", "unknown"),
            score
        )

        return {
            "parent_score": round(score, 4),
            "explanation": explanation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# -------------------------
# UTILITIES
# -------------------------

def download_file(url: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    r = requests.get(url, stream=True, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to download {url}")
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)


def safe_gemini_explain(career_id: str, score: float) -> str:
    if llm is None:
        return "Explanation unavailable."

    try:
        prompt = (
            f"Explain in 1–2 sentences why parents might prefer the career "
            f"'{career_id}' given a compatibility score of {score:.2f}."
        )
        resp = llm.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return resp.text.strip()
    except Exception:
        return "This career aligns well with parental priorities such as stability, finances, and long-term prospects."


# -------------------------
# STARTUP
# -------------------------

@app.on_event("startup")
def startup():
    global model, preprocessor, careers_df, llm

    # 1. Download artifacts
    download_file(MODEL_URL, MODEL_PATH)
    download_file(PREPROC_URL, PREPROC_PATH)

    # 2. Load artifacts
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROC_PATH)

    # 3. Load careers metadata
    careers_df = pd.read_csv(CAREERS_CSV)

    # 4. Init Gemini (optional but preferred)
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        llm = genai.Client(api_key=api_key)
    else:
        llm = None


# -------------------------
# SCHEMA
# -------------------------

class ParentRequest(BaseModel):
    budget_max_tuition: float

    importance_finances: int
    importance_job_security: int
    importance_prestige: int
    parent_risk_tolerance: int
    influence_from_people: int

    migration_allowed: bool
    location_preference: str

    unacceptable_careers: List[str] = []


# -------------------------
# ENDPOINT
# -------------------------

@app.post("/rescore-parent")
def rescore_parent(req: ParentRequest):
    if model is None or preprocessor is None:
        raise HTTPException(500, "Model not loaded")

    df = careers_df.copy()

    # Parent features (broadcast)
    df["budget_max_tuition"] = req.budget_max_tuition
    df["importance_finances"] = req.importance_finances
    df["importance_job_security"] = req.importance_job_security
    df["importance_prestige"] = req.importance_prestige
    df["parent_risk_tolerance"] = req.parent_risk_tolerance
    df["influence_from_people"] = req.influence_from_people
    df["migration_allowed"] = int(req.migration_allowed)
    df["location_preference"] = req.location_preference
    df["is_unacceptable"] = df["career_id"].isin(req.unacceptable_careers).astype(int)

    X = preprocessor.transform(df)
    scores = model.predict(X)

    df["parent_score"] = scores
    df = df.sort_values("parent_score", ascending=False)

    top5 = df.head(5)[["career_id", "parent_score"]].to_dict(orient="records")
    best = top5[0]

    explanation = safe_gemini_explain(best["career_id"], best["parent_score"])

    return {
        "top_5_parent_scores": top5,
        "final_recommendation": {
            "career_id": best["career_id"],
            "parent_score": round(best["parent_score"], 3),
            "parent_explanation": explanation
        }
    }