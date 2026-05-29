import json
import os
import time
from typing import Optional

import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.service.observability import (
    FRAUD_COUNT,
    MODEL_INFO,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    setup_logging,
)

load_dotenv()

MODEL_PATH        = os.getenv("MODEL_PATH",        "models/final_model.pkl")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "models/preprocessor.pkl")
THRESHOLD_PATH    = os.getenv("THRESHOLD_PATH",    "models/threshold.json")
API_KEY           = os.getenv("API_KEY", "")

logger = setup_logging(os.getenv("LOG_LEVEL", "INFO"))


def _load_artifacts():
    model        = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    with open(THRESHOLD_PATH) as f:
        data = json.load(f)
    threshold = float(data["threshold"])
    MODEL_INFO.set(threshold)
    logger.info("artifacts loaded, threshold=%.4f", threshold)
    return model, preprocessor, threshold


model, preprocessor, THRESHOLD = _load_artifacts()

app = FastAPI(
    title="Fraud Detection API",
    description="Определяем мошеннические транзакции по кредитным картам.",
    version="1.0.0",
)


class Transaction(BaseModel):
    Time:   float = Field(..., description="Секунды от первой транзакции в датасете")
    Amount: float = Field(..., description="Сумма транзакции в EUR")
    V1:  float = 0.0; V2:  float = 0.0; V3:  float = 0.0; V4:  float = 0.0
    V5:  float = 0.0; V6:  float = 0.0; V7:  float = 0.0; V8:  float = 0.0
    V9:  float = 0.0; V10: float = 0.0; V11: float = 0.0; V12: float = 0.0
    V13: float = 0.0; V14: float = 0.0; V15: float = 0.0; V16: float = 0.0
    V17: float = 0.0; V18: float = 0.0; V19: float = 0.0; V20: float = 0.0
    V21: float = 0.0; V22: float = 0.0; V23: float = 0.0; V24: float = 0.0
    V25: float = 0.0; V26: float = 0.0; V27: float = 0.0; V28: float = 0.0

    model_config = {"json_schema_extra": {
        "example": {
            "Time": 0.0, "Amount": 149.62,
            "V1": -1.36, "V2": -0.07, "V3": 2.54,  "V4": 1.38,
            "V5": -0.34, "V6":  0.46, "V7": 0.24,  "V8": 0.10,
            "V9":  0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
            "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
            "V17": 0.21, "V18": 0.03,  "V19": 0.40, "V20": 0.25,
            "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
            "V25": 0.13,  "V26": -0.19, "V27": 0.13, "V28": -0.02,
        }
    }}


class PredictResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float


def _check_api_key(key: Optional[str]) -> None:
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


@app.get("/health")
def health():
    return {"status": "ok", "model": "LightGBM", "threshold": THRESHOLD}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(tx: Transaction, x_api_key: Optional[str] = Header(default=None)):
    _check_api_key(x_api_key)
    t0 = time.perf_counter()
    try:
        import pandas as pd
        row = pd.DataFrame([{
            "Time": tx.Time, "Amount": tx.Amount,
            **{f"V{i}": getattr(tx, f"V{i}") for i in range(1, 29)},
        }])
        X    = preprocessor.transform(row)
        prob = float(model.predict_proba(X)[0, 1])
        flag = prob >= THRESHOLD

        FRAUD_COUNT.labels(decision="fraud" if flag else "legit").inc()
        REQUEST_COUNT.labels(endpoint="/predict", status="200").inc()
        logger.info("predict amount=%.2f prob=%.4f result=%s", tx.Amount, prob, "fraud" if flag else "legit")

        return PredictResponse(fraud_probability=round(prob, 6), is_fraud=bool(flag), threshold=THRESHOLD)
    except Exception as exc:
        REQUEST_COUNT.labels(endpoint="/predict", status="500").inc()
        logger.error("predict failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.perf_counter() - t0)
