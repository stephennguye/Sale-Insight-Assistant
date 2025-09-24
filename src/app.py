from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.exceptions import ModelTrainingError, RAGError
from src.analytics import compute_kpis
from src.rag_query import generate_answer

logger = get_logger("api")

MODEL_PATH = Path("models/churn_xgb.pkl")
FEATURES_PATH = Path("models/churn_features.pkl")

app = FastAPI(title="Sales Insights Assistant")

# Lazy-loaded objects
_model = None
_features = None


def _load_model_and_features():
    global _model, _features
    if _model is None or _features is None:
        if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
            raise ModelTrainingError("Model or features not found. Run training first.")
        _model = joblib.load(MODEL_PATH)
        _features = joblib.load(FEATURES_PATH)
    return _model, _features


@app.get("/kpis")
def kpis():
    try:
        rows = compute_kpis()
        return {r['metric']: r['value'] for r in rows.to_dict(orient='records')}
    except Exception as e:
        logger.exception("Failed to fetch KPIs")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(payload: dict):
    try:
        model, features = _load_model_and_features()
        X = pd.DataFrame([payload])
        X = pd.get_dummies(X)

        # Align with training columns
        X = X.reindex(columns=features, fill_value=0)

        prob = float(model.predict_proba(X)[0][1])
        return {"churn_prob": prob}
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/ask")
def ask(query: str):
    try:
        ans = generate_answer(query)
        return {"answer": ans}
    except RAGError as e:
        logger.exception("RAG error")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unknown error in /ask")
        raise HTTPException(status_code=500, detail=str(e))
