from __future__ import annotations

import pickle
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.core.logger import get_logger
from app.ml.base import BaseMLModel
from app.ml.factory import MLModelFactory

logger = get_logger("LightGBMModel")


class LightGBMModel(BaseMLModel):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 8,
        learning_rate: float = 0.1,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._model = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "lightgbm"

    @property
    def model_type(self) -> str:
        return "classification"

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        logger.info(f"Training LightGBM model with {X.shape[0]} samples")
        self._model.fit(X, y)
        self._is_trained = True
        self._feature_names = feature_names or []

        train_score = self._model.score(X, y)
        logger.info(f"Training accuracy: {train_score:.4f}")

        return {"train_accuracy": train_score}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)[:, 1]
        metrics: dict[str, float] = {
            "accuracy": accuracy_score(y, predictions),
            "precision": precision_score(y, predictions, zero_division=0),
            "recall": recall_score(y, predictions, zero_division=0),
            "f1": f1_score(y, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y, probabilities),
            "pr_auc": average_precision_score(y, probabilities),
            "confusion_matrix": confusion_matrix(y, predictions),
            "probabilities": probabilities,
        }
        return metrics

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        self._is_trained = True
        logger.info(f"Model loaded from {path}")

    def get_feature_importance(self) -> dict[str, float]:
        if not self._is_trained:
            return {}

        importances = self._model.feature_importances_
        if self._feature_names:
            return dict(zip(self._feature_names, importances))
        return {f"feature_{i}": imp for i, imp in enumerate(importances)}


MLModelFactory.register("lightgbm", LightGBMModel)
