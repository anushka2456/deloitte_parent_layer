from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from pathlib import Path
import pandas as pd
import joblib
import os

# ---------- Gemini (stable SDK) ----------
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------- Paths ----------
BASE_DIR = Path(__file__).parent

DATA_PATH = BASE_DIR / "parent_only_synthetic_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "parent_layer" / "parent_model_rf.joblib"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessors" / "parent_preprocessor_coltransformer.joblib"

# ---------- Load artifacts (OLD BEHAVIOR) ----------
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# ---------- Load careers ----------
careers_df = (
    pd.read_csv(DATA_PATH)[
        [
            "career_id",
            "c_avg_salary",
            "c_job_security",
            "c_prestige",
            "c_tuition",
            "c_location"
        ]
    ]
    .drop_duplicates("career_id")
)

# ---------- FastAPI ----------
app = FastAPI(
    title="Parent Layer Career Recommendation API",
    version="1.0"
)

# ---------- Input Schema ----------
class ParentInput(BaseModel):
    budget_max_tuition: float = Field(..., gt=0)

    importance_finances: int = Field(..., ge=1, le=5)
    importance_job_security: int = Field(..., ge=1, le=5)
    importance_prestige: int = Field(..., ge=1, le=5)
    parent_risk_tolerance: int = Field(..., ge=1, le=5)
    influence_from_people: int = Field(..., ge=1, le=5)

    location_preference: str = Field(..., pattern="^(local|national|international)$")
    migration_allowed: bool

    unacceptable_careers: List[str] = []

# ---------- Helpers ----------
def normalize_likert(x: int) -> float:
    return (x - 1) / 4.0


def location_match(parent_pref: str, career_loc: str) -> int:
    order = {"local": 0, "national": 1, "international": 2}
    return int(order[career_loc] <= order[parent_pref])


def explain_with_gemini(career_id: str, score: float) -> str:
    prompt = f"""
Explain to a parent why the career '{career_id}' received a high recommendation score of {round(score, 2)}.

Focus on:
- job security
- income stability
- prestige
- long-term prospects

Use simple language. Avoid technical terms.
"""
    llm = genai.GenerativeModel("gemini-2.5-pro")
    response = llm.generate_content(prompt)
    return response.text.strip()

# ---------- Endpoint ----------
@app.post("/rescore-parent")
def rescore_parent(input: ParentInput):

    parent_features = {
        "p_financial_weight": normalize_likert(input.importance_finances),
        "p_job_security_weight": normalize_likert(input.importance_job_security),
        "p_prestige_weight": normalize_likert(input.importance_prestige),
        "p_parent_risk_tolerance": normalize_likert(input.parent_risk_tolerance),
        "p_weight_on_parent_layer": normalize_likert(input.influence_from_people),
        "p_budget_max_tuition": input.budget_max_tuition,
    }

    rows = []
    career_ids = []

    for _, c in careers_df.iterrows():
        rows.append({
            **parent_features,

            "c_avg_salary": c.c_avg_salary,
            "c_job_security": c.c_job_security,
            "c_prestige": c.c_prestige,
            "c_tuition": c.c_tuition,

            "f_fin_ratio": c.c_avg_salary / max(1, input.budget_max_tuition * 3),
            "f_tuition_affordable": int(c.c_tuition <= input.budget_max_tuition),
            "f_location_match": location_match(input.location_preference, c.c_location),
            "f_migration_ok": int(input.migration_allowed or c.c_location != "international"),
            "f_is_unacceptable": int(c.career_id in input.unacceptable_careers),
            "f_risk_penalty": 1.0
        })
        career_ids.append(c.career_id)

    df = pd.DataFrame(rows)
    X = preprocessor.transform(df)
    scores = model.predict(X)

    results = [
        {
            "career_id": career_ids[i],
            "parent_score": round(float(scores[i]), 3)
        }
        for i in range(len(scores))
    ]

    results.sort(key=lambda x: x["parent_score"], reverse=True)

    top_5 = results[:5]
    best = top_5[0]

    try:
        explanation = explain_with_gemini(best["career_id"], best["parent_score"])
    except Exception:
        explanation = (
            "This career aligns strongly with parental priorities such as financial stability, "
            "job security, and long-term prospects."
        )

    return {
        "top_5_parent_scores": top_5,
        "final_recommendation": {
            "career_id": best["career_id"],
            "parent_score": best["parent_score"],
            "parent_explanation": explanation
        }
    }
