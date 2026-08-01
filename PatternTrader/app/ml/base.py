from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class MLPrediction(BaseModel):
    model_name: str
    symbol: str
    timeframe: str
    pattern_name: str
    probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    features_used: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_high_probability(self) -> bool:
        return self.probability >= 0.7

    @property
    def is_actionable(self) -> bool:
        return self.probability >= 0.75 and self.confidence >= 0.6


class BaseMLModel(ABC):
    """Base class for all ML models."""

    def __init__(self) -> None:
        self._is_trained = False
        self._feature_names: list[str] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name."""
        ...

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Model type (classification, regression, etc.)."""
        ...

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @abstractmethod
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train the model."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        ...

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Evaluate model performance."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        ...

    def get_prediction(
        self,
        features: np.ndarray,
        symbol: str,
        timeframe: str,
        pattern_name: str,
    ) -> MLPrediction:
        """Get a formatted prediction."""
        if not self._is_trained:
            raise ValueError("Model is not trained yet")

        probability = float(self.predict_proba(features.reshape(1, -1))[0][1])
        confidence = self._calculate_confidence(features)

        return MLPrediction(
            model_name=self.name,
            symbol=symbol,
            timeframe=timeframe,
            pattern_name=pattern_name,
            probability=probability,
            confidence=confidence,
            features_used=self._feature_names,
        )

    def _calculate_confidence(self, features: np.ndarray) -> float:
        """Calculate prediction confidence (to be overridden)."""
        return 0.7

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance (to be overridden)."""
        return {}
