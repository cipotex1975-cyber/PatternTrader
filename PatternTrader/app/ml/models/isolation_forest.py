from __future__ import annotations

import pickle
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, roc_auc_score

from app.core.logger import get_logger
from app.ml.base import BaseMLModel
from app.ml.factory import MLModelFactory

logger = get_logger("IsolationForestModel")


class IsolationForestModel(BaseMLModel):
    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.05,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            **kwargs,
        )
        self._contamination = contamination

    @property
    def name(self) -> str:
        return "isolation_forest"

    @property
    def model_type(self) -> str:
        return "anomaly"

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        logger.info(f"Training Isolation Forest with {X.shape[0]} samples")
        self._model.fit(X)
        self._is_trained = True
        self._feature_names = feature_names or []

        outliers = int((self._model.predict(X) == -1).sum())
        return {"samples": int(X.shape[0]), "outliers": outliers}

    def _anomaly_proba(self, X: np.ndarray) -> np.ndarray:
        decisions = self._model.decision_function(X)
        return 1.0 / (1.0 + np.exp(decisions))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self._anomaly_proba(X) > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self._anomaly_proba(X)
        return np.column_stack([1.0 - p, p])

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)[:, 1]

        labels = np.asarray(y)
        if set(np.unique(labels)).issubset({-1, 1}):
            labels = (labels == -1).astype(int)

        metrics: dict[str, float] = {
            "accuracy": accuracy_score(labels, predictions),
        }
        if len(np.unique(labels)) > 1:
            metrics["roc_auc"] = roc_auc_score(labels, probabilities)
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


MLModelFactory.register("isolation_forest", IsolationForestModel)
