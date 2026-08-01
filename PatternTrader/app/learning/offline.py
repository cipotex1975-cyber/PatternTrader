from __future__ import annotations

from typing import Any, Optional

import numpy as np
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from app.backtesting.metrics import MetricsCalculator
from app.core.logger import get_logger
from app.learning.features import FeatureBuilder
from app.learning.models import KnowledgeEntry

logger = get_logger("OfflineLearner")


class OfflineLearner:
    """Aprendizaje offline: entrena un modelo sobre toda la base de conocimiento
    y valida con validación cruzada estratificada."""

    def __init__(
        self,
        feature_builder: Optional[FeatureBuilder] = None,
        model_path: Optional[str] = None,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self._feature_builder = feature_builder or FeatureBuilder()
        self._model_path = model_path
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        self._is_trained = False
        self._last_report: dict[str, Any] = {}

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(
        self,
        entries: list[KnowledgeEntry],
        n_splits: int = 5,
        test_size: float = 0.25,
    ) -> dict[str, Any]:
        X, y, feature_names = self._feature_builder.matrix(entries)
        if len(y) < 2:
            return {
                "trained": False,
                "samples": len(y),
                "error": "Se necesitan al menos 2 operaciones para entrenar",
            }

        self._model.fit(X, y)
        self._is_trained = True

        if len(np.unique(y)) < 2 or len(y) < max(2, n_splits):
            cv_report: dict[str, Any] = {"skipped": "Clase única o pocas muestras"}
        else:
            cv_report = self._cross_validate(X, y, feature_names, n_splits)

        train_pred = self._model.predict(X)
        train_metrics = {
            "accuracy": float(accuracy_score(y, train_pred)),
            "precision": float(precision_score(y, train_pred, zero_division=0)),
            "recall": float(recall_score(y, train_pred, zero_division=0)),
            "f1": float(f1_score(y, train_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y, train_pred).tolist(),
        }

        report = {
            "trained": True,
            "samples": int(len(y)),
            "wins": int(y.sum()),
            "losses": int(len(y) - y.sum()),
            "feature_names": feature_names,
            "feature_importance": self._feature_importance(feature_names),
            "cross_validation": cv_report,
            "train_metrics": train_metrics,
            "model_path": self._model_path,
        }
        self._last_report = report

        if self._model_path:
            dump(self._model, self._model_path)

        logger.info(f"Offline training completado: {len(y)} muestras")
        return report

    def _cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        n_splits: int,
    ) -> dict[str, Any]:
        safe_splits = max(1, min(n_splits, int(y.sum()), int(len(y) - y.sum())))
        cv = StratifiedKFold(n_splits=safe_splits, shuffle=True, random_state=42)
        fold_metrics: list[dict[str, float]] = []
        all_true: list[int] = []
        all_pred: list[int] = []
        all_proba: list[float] = []

        for train_idx, test_idx in cv.split(X, y):
            self._model.fit(X[train_idx], y[train_idx])
            proba = self._model.predict_proba(X[test_idx])[:, 1]
            pred = (proba >= 0.5).astype(int)
            all_true.extend(y[test_idx].tolist())
            all_pred.extend(pred.tolist())
            all_proba.extend(proba.tolist())
            fold_metrics.append(
                {
                    "precision": float(precision_score(y[test_idx], pred, zero_division=0)),
                    "recall": float(recall_score(y[test_idx], pred, zero_division=0)),
                    "f1": float(f1_score(y[test_idx], pred, zero_division=0)),
                }
            )

        cm = MetricsCalculator.classification_metrics(
            all_true, all_pred, all_proba
        )
        return {
            "n_splits": len(fold_metrics),
            "per_fold": fold_metrics,
            "average_precision": float(np.mean([f["precision"] for f in fold_metrics])),
            "average_recall": float(np.mean([f["recall"] for f in fold_metrics])),
            "average_f1": float(np.mean([f["f1"] for f in fold_metrics])),
            "precision": cm.precision,
            "recall": cm.recall,
            "f1": cm.f1_score,
            "roc_auc": cm.roc_auc,
            "pr_auc": cm.pr_auc,
            "confusion_matrix": cm.confusion_matrix,
        }

    def predict_proba(self, features: list[float]) -> float:
        if not self._is_trained:
            raise ValueError("El modelo offline no está entrenado")
        X = np.array([features], dtype=float)
        return float(self._model.predict_proba(X)[0][1])

    def predict(self, features: list[float]) -> int:
        return 1 if self.predict_proba(features) >= 0.5 else 0

    def _feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        importances = self._model.feature_importances_
        return {
            name: float(imp)
            for name, imp in sorted(
                zip(feature_names, importances), key=lambda kv: kv[1], reverse=True
            )
        }

    def load(self, path: str) -> None:
        self._model = load(path)
        self._is_trained = True
        logger.info(f"Modelo offline cargado desde {path}")

    def save(self, path: str) -> None:
        dump(self._model, path)
        self._model_path = path
        logger.info(f"Modelo offline guardado en {path}")

    @property
    def last_report(self) -> dict[str, Any]:
        return dict(self._last_report)
