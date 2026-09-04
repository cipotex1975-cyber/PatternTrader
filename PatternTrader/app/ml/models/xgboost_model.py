from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from app.core.logger import get_logger
from app.ml.base import BaseMLModel
from app.ml.factory import MLModelFactory

logger = get_logger("XGBoostModel")


class XGBoostModel(BaseMLModel):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
        early_stopping_rounds: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._early_stopping_rounds = early_stopping_rounds
        xgb_kwargs: dict[str, Any] = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="logloss",
            random_state=random_state,
        )
        # xgboost>=2: `early_stopping_rounds` es param del CONSTRUCTOR (no de fit).
        xgb_kwargs.update(kwargs)
        if early_stopping_rounds > 0:
            xgb_kwargs["early_stopping_rounds"] = early_stopping_rounds
        self._model = XGBClassifier(**xgb_kwargs)

    @property
    def name(self) -> str:
        return "xgboost"

    @property
    def model_type(self) -> str:
        return "classification"

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        logger.info(f"Training XGBoost model with {X.shape[0]} samples")
        has_validation = X_val is not None and y_val is not None
        if has_validation:
            self._model.fit(
                X,
                y,
                eval_set=[(np.asarray(X_val), np.asarray(y_val))],
                verbose=False,
            )
        else:
            self._model.fit(X, y)

        self._is_trained = True
        self._feature_names = feature_names or []

        train_score = self._model.score(X, y)
        logger.info(f"Training accuracy: {train_score:.4f}")

        result: dict[str, Any] = {"train_accuracy": train_score, "early_stopping": False}
        if has_validation:
            val_score = self._model.score(np.asarray(X_val), np.asarray(y_val))
            best_iteration = getattr(self._model, "best_iteration", None)
            result["validation_accuracy"] = float(val_score)
            result["early_stopping"] = self._early_stopping_rounds > 0
            if best_iteration is not None:
                result["best_iteration"] = int(best_iteration)
            logger.info(
                f"Validation accuracy: {val_score:.4f}"
                + (f" | best_iteration={int(best_iteration)}" if best_iteration is not None else "")
            )

        return result

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
        self._model.save_model(path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        self._model.load_model(path)
        self._is_trained = True
        logger.info(f"Model loaded from {path}")

    def get_feature_importance(self) -> dict[str, float]:
        if not self._is_trained:
            return {}

        importances = self._model.feature_importances_
        if self._feature_names:
            return dict(zip(self._feature_names, importances))
        return {f"feature_{i}": imp for i, imp in enumerate(importances)}


MLModelFactory.register("xgboost", XGBoostModel)
