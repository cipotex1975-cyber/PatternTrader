from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import SGDClassifier

from app.core.logger import get_logger
from app.learning.features import FeatureBuilder

logger = get_logger("OnlineLearner")


class OnlineLearner:
    """Aprendizaje online: actualización incremental tras cada operación cerrada."""

    def __init__(
        self,
        feature_builder: Optional[FeatureBuilder] = None,
        learning_rate: float = 0.001,
        loss: str = "modified_huber",
    ) -> None:
        self._feature_builder = feature_builder or FeatureBuilder()
        self._clf = SGDClassifier(
            loss=loss,
            learning_rate="adaptive",
            eta0=learning_rate,
            random_state=42,
        )
        self._n_samples = 0
        self._seen = 0

    @property
    def is_trained(self) -> bool:
        return self._n_samples > 0

    @property
    def samples_seen(self) -> int:
        return self._n_samples

    def update(self, features: list[float], outcome: int) -> None:
        """Aprende de una única operación (WIN=1, LOSS=0)."""
        X = np.array([features], dtype=float)
        y = np.array([int(outcome)], dtype=int)
        if self._n_samples == 0:
            self._clf.partial_fit(X, y, classes=np.array([0, 1]))
        else:
            self._clf.partial_fit(X, y)
        self._n_samples += 1
        self._seen += 1

    def update_batch(self, features_matrix: np.ndarray, outcomes: np.ndarray) -> None:
        for features, outcome in zip(features_matrix, outcomes):
            self.update(features, outcome)

    def predict_proba(self, features: list[float]) -> float:
        if not self.is_trained:
            return 0.5
        X = np.array([features], dtype=float)
        proba = self._clf.predict_proba(X)
        return float(proba[0][1])

    def predict(self, features: list[float]) -> int:
        if not self.is_trained:
            return 0
        X = np.array([features], dtype=float)
        return int(self._clf.predict(X)[0])
