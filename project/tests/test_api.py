import sys
sys.path.insert(0, '.')

import os
os.environ.setdefault("MODEL_PATH",        "models/final_model.pkl")
os.environ.setdefault("PREPROCESSOR_PATH", "models/preprocessor.pkl")
os.environ.setdefault("THRESHOLD_PATH",    "artifacts/threshold.json")
os.environ.setdefault("API_KEY",           "")

from fastapi.testclient import TestClient
from src.service.main import app

client = TestClient(app)

SAMPLE_TX = {
    "Time": 0.0, "Amount": 149.62,
    "V1": -1.36, "V2": -0.07, "V3":  2.54, "V4":  1.38,
    "V5": -0.34, "V6":  0.46, "V7":  0.24, "V8":  0.10,
    "V9":  0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17":  0.21, "V18": 0.03, "V19":  0.40, "V20":  0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24":  0.07,
    "V25":  0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "threshold" in r.json()


def test_predict_200():
    assert client.post("/predict", json=SAMPLE_TX).status_code == 200


def test_predict_schema():
    body = client.post("/predict", json=SAMPLE_TX).json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)
    assert "threshold" in body


def test_predict_missing_field():
    bad = {k: v for k, v in SAMPLE_TX.items() if k != "Amount"}
    assert client.post("/predict", json=bad).status_code == 422


def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "fraud_api_requests_total" in r.text


def test_api_key_auth():
    os.environ["API_KEY"] = "secret123"
    import importlib, src.service.main as svc
    importlib.reload(svc)
    c = TestClient(svc.app)

    assert c.post("/predict", json=SAMPLE_TX, headers={"x-api-key": "wrong"}).status_code == 403
    assert c.post("/predict", json=SAMPLE_TX, headers={"x-api-key": "secret123"}).status_code == 200

    os.environ["API_KEY"] = ""
