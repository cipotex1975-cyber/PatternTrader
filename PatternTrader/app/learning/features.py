from __future__ import annotations

from typing import Any, Optional

import numpy as np

from app.core.logger import get_logger
from app.learning.models import KnowledgeEntry
from app.ml.features import TECHNICAL_FEATURE_NAMES

logger = get_logger("FeatureBuilder")

# El modelo de conocimiento usa el mismo vector que el ScoringEngine para que
# ambos modelos sean intercambiables (unificación de features).
DEFAULT_FEATURES = list(TECHNICAL_FEATURE_NAMES)


class FeatureBuilder:
    """Convierte indicadores/variables de una operación en un vector de features."""

    def __init__(self, feature_names: Optional[list[str]] = None) -> None:
        self._feature_names = feature_names or list(DEFAULT_FEATURES)

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def build(self, indicators: dict[str, Any], variables: dict[str, Any]) -> list[float]:
        combined: dict[str, Any] = {}
        combined.update(indicators or {})
        combined.update(variables or {})
        return [float(combined.get(name, 0.0) or 0.0) for name in self._feature_names]

    def entry_vector(self, entry: KnowledgeEntry) -> list[float]:
        if entry.ml_features:
            return list(entry.ml_features)
        return self.build(entry.indicators, entry.variables)

    def matrix(
        self,
        entries: list[KnowledgeEntry],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Construye (X, y, feature_names) a partir de la base de conocimiento."""
        if not entries:
            empty = np.empty((0, len(self._feature_names)), dtype=float)
            return empty, np.empty((0,), dtype=int), self._feature_names

        rows = []
        labels = []
        for entry in entries:
            features = self.entry_vector(entry)
            if len(features) != len(self._feature_names):
                features = self.build(entry.indicators, entry.variables)
            rows.append(features)
            labels.append(entry.is_win)

        X = np.asarray(rows, dtype=float)
        y = np.asarray(labels, dtype=int)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, y, self._feature_names
