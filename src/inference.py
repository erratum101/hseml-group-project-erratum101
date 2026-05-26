from __future__ import annotations

import os
from typing import Any

import joblib
import pandas as pd

from config import FEATURE_COLUMNS

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")


class ModelNotLoadedError(RuntimeError):
    pass


class Predictor:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
        self._model: Any | None = None
        self._model_name: str = "unknown"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if not os.path.isfile(self.model_path):
            raise ModelNotLoadedError(
                f"Модель не найдена: {self.model_path}. "
                "Выполните `make train` или `python scripts/train_demo_model.py`."
            )
        self._model = joblib.load(self.model_path)
        self._model_name = type(self._model).__name__
        if hasattr(self._model, "named_steps"):
            self._model_name = type(self._model.named_steps.get("clf", self._model)).__name__

    def _ensure_loaded(self) -> Any:
        if self._model is None:
            self.load()
        return self._model

    def predict(self, features: dict[str, float | int]) -> dict[str, float | int | bool | str]:
        model = self._ensure_loaded()
        row = pd.DataFrame([{col: features[col] for col in FEATURE_COLUMNS}])
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(row)[0, 1])
        else:
            pred = int(model.predict(row)[0])
            proba = float(pred)
        label = int(proba >= 0.5)
        return {
            "dissatisfied": bool(label),
            "probability": proba,
            "label": label,
            "model_name": self._model_name,
        }


_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor
