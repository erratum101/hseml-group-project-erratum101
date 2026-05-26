import os
import sys

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import FEATURE_COLUMNS, RANDOM_STATE  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    model_path = tmp_path / "best_model.pkl"
    X = np.random.default_rng(RANDOM_STATE).uniform(0, 1, size=(100, len(FEATURE_COLUMNS)))
    y = (X[:, 6] > 0.5).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=RANDOM_STATE)
    model.fit(X, y)
    joblib.dump(model, model_path)

    monkeypatch.setenv("MODEL_PATH", str(model_path))

    import importlib

    import inference

    inference._predictor = None
    import api

    importlib.reload(api)
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"


def test_predict(client):
    payload = {
        "payment_value": 100.0,
        "price": 80.0,
        "freight_value": 20.0,
        "items_count": 1,
        "payment_installments": 1,
        "delivery_time_days": 10.0,
        "delivery_delay_days": 3.0,
        "review_text_length": 50,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "probability" in body
    assert body["label"] in (0, 1)
    assert isinstance(body["dissatisfied"], bool)


def test_example_endpoint(client):
    r = client.get("/example")
    assert r.status_code == 200
    assert set(r.json().keys()) == set(FEATURE_COLUMNS)
