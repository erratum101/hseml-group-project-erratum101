"""Обучает небольшую модель на синтетике, если нет Kaggle-данных (для демо API)."""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import FEATURE_COLUMNS, RANDOM_STATE  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "models", "best_model.pkl")


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    n = 2000
    X = pd.DataFrame(
        rng.uniform(0, 1, size=(n, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )
    y = ((X["delivery_time_days"] > 0.6) | (X["delivery_delay_days"] > 0.5)).astype(int)
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X, y)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Demo model saved to {MODEL_PATH}")
    print("Note: for real metrics run `make preprocess && make train`.")


if __name__ == "__main__":
    main()
