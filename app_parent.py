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
