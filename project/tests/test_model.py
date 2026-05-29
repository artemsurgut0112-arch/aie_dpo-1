import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
import pytest
import joblib

MODEL_PATH        = "models/final_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
THRESHOLD_PATH    = "models/threshold.json"


@pytest.fixture(scope="module")
def artifacts():
    model        = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    with open(THRESHOLD_PATH) as f:
        data = json.load(f)
    return model, preprocessor, float(data["threshold"])


def test_artifacts_load(artifacts):
    model, preprocessor, threshold = artifacts
    assert model is not None
    assert preprocessor is not None
    assert 0.0 < threshold < 1.0


def test_predict_proba_shape(artifacts):
    model, preprocessor, _ = artifacts
    row = pd.DataFrame([{"Time": 0.0, "Amount": 100.0, **{f"V{i}": 0.0 for i in range(1, 29)}}])
    proba = model.predict_proba(preprocessor.transform(row))
    assert proba.shape == (1, 2)
    assert abs(proba[0].sum() - 1.0) < 1e-6


def test_predict_proba_range(artifacts):
    model, preprocessor, _ = artifacts
    rows = pd.DataFrame([
        {"Time": float(i), "Amount": float(i * 10), **{f"V{j}": np.random.randn() for j in range(1, 29)}}
        for i in range(20)
    ])
    probas = model.predict_proba(preprocessor.transform(rows))[:, 1]
    assert (probas >= 0).all() and (probas <= 1).all()


def test_threshold_json(artifacts):
    _, _, threshold = artifacts
    with open(THRESHOLD_PATH) as f:
        data = json.load(f)
    assert "LightGBM" in data["model"]
    assert abs(data["threshold"] - threshold) < 1e-9
