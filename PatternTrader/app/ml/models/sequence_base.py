from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from app.core.logger import get_logger
from app.ml.base import BaseMLModel, MLPrediction

logger = get_logger("SequenceModel")


def _early_stopping_decision(val_history: list[float], patience: int) -> tuple[int, int]:
    """Decisión de early stopping dada la serie de validation loss (FASE 6).

    Devuelve ``(best_epoch, epochs_trained)`` en números de epoch (1-based):

    - ``best_epoch`` es la posición del mínimo de ``val_history``.
    - ``epochs_trained`` es ``best_epoch + patience`` (el entrenamiento se
      detiene tras ``patience`` epochs consecutivos sin mejorar).

    Ejemplo (patience=5): si el mínimo cae en el epoch 12, el entrenamiento
    para en el epoch 17. Corresponde al ejemplo de la especificación FASE 6.
    """
    if not val_history:
        return 0, 0
    best_idx = int(np.argmin(val_history))
    best_epoch = best_idx + 1
    epochs_trained = min(len(val_history), best_epoch + patience)
    return best_epoch, epochs_trained


class _SequenceClassifier(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(backbone.output_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.backbone(x))


class SequenceModel(BaseMLModel):
    """Base class for time-series classification models (torch)."""

    def __init__(
        self,
        sequence_length: int = 30,
        feature_dim: int = 1,
        hidden_dim: int = 64,
        epochs: int = 10,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        random_state: int = 42,
        patience: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._sequence_length = sequence_length
        self._feature_dim = feature_dim
        self._hidden_dim = hidden_dim
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._random_state = random_state
        self._patience = patience
        self._best_epoch = 0
        self._best_validation_loss: float | None = None
        self._model: nn.Module | None = None
        self._scaler = None  # StandardScaler for scaling sequence data

    @property
    def model_type(self) -> str:
        return "sequence_classification"

    def _build_network(self) -> nn.Module:
        raise NotImplementedError

    def _ensure_model(self) -> nn.Module:
        if self._model is None:
            torch.manual_seed(self._random_state)
            self._model = _SequenceClassifier(self._build_network())
        return self._model

    def _config(self) -> dict[str, Any]:
        return {
            "sequence_length": self._sequence_length,
            "feature_dim": self._feature_dim,
            "hidden_dim": self._hidden_dim,
            "epochs": self._epochs,
            "learning_rate": self._learning_rate,
            "batch_size": self._batch_size,
            "random_state": self._random_state,
            "patience": self._patience,
            "best_epoch": self._best_epoch,
            "best_validation_loss": self._best_validation_loss,
        }

    def _apply_config(self, config: dict[str, Any]) -> None:
        self._sequence_length = config.get("sequence_length", self._sequence_length)
        self._feature_dim = config.get("feature_dim", self._feature_dim)
        self._hidden_dim = config.get("hidden_dim", self._hidden_dim)
        self._epochs = config.get("epochs", self._epochs)
        self._learning_rate = config.get("learning_rate", self._learning_rate)
        self._batch_size = config.get("batch_size", self._batch_size)
        self._random_state = config.get("random_state", self._random_state)
        self._patience = config.get("patience", self._patience)
        self._best_epoch = config.get("best_epoch", self._best_epoch)
        self._best_validation_loss = config.get("best_validation_loss", self._best_validation_loss)

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            # FASE 4: aplicar el scaler fit en TRAIN (si existe) sobre features
            # crudas antes de construir la secuencia (raw → scaler → sequence).
            if self._scaler is not None and X.shape[1] == self._feature_dim:
                X = self._scaler.transform(X).astype(np.float32)
            if X.shape[1] == self._sequence_length * self._feature_dim:
                X = X.reshape(X.shape[0], self._sequence_length, self._feature_dim)
            else:
                X = X.reshape(X.shape[0], X.shape[1], 1)
        return X

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Entrena con early stopping real sobre VALIDATION (FASE 6).

        - ``X_val``/``y_val`` (opcional): bloque VALIDATION usado durante el
          entrenamiento; NUNCA el TEST FINAL.
        - Si se pasa validación, se entrena epoch a epoch midiendo
          ``validation_loss``; con ``self._patience`` epochs sin mejora se
          detiene (early stopping) y se restaura el MEJOR estado (checkpoint).
        - Reporta ``train_loss``, ``validation_loss``, ``train_accuracy`` y
          ``validation_accuracy`` por epoch cuando están disponibles.
        """
        X = self._prepare(X)
        if X.ndim != 3:
            raise ValueError(
                f"Sequence model expects 3D input (samples, timesteps, features), got {X.shape}"
            )

        self._feature_names = feature_names or []
        model = self._ensure_model()
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self._learning_rate)
        criterion = nn.CrossEntropyLoss()

        dataset = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.int64))
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        has_validation = X_val is not None and y_val is not None
        val_loader: DataLoader | None = None
        if has_validation:
            assert X_val is not None and y_val is not None
            X_val = self._prepare(X_val)
            if X_val.ndim != 3:
                raise ValueError(
                    "Sequence model validation expects 3D input "
                    f"(samples, timesteps, features), got {X_val.shape}"
                )
            val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.int64))
            val_loader = DataLoader(val_dataset, batch_size=self._batch_size, shuffle=False)

        best_val_loss = float("inf")
        best_state: dict[str, Any] | None = None
        val_history: list[float] = []
        no_improve_streak = 0
        stopped_early = False

        last_train_loss = float("nan")
        last_train_acc = float("nan")
        last_val_loss = float("nan")
        last_val_acc = float("nan")

        for epoch in range(1, self._epochs + 1):
            # --- Train: un epoch ---
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            correct = 0
            total = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
                preds = model(xb).argmax(dim=1)
                correct += int(torch.eq(preds, yb).sum().item())
                total += int(yb.numel())

            last_train_loss = epoch_loss / max(1, num_batches)
            last_train_acc = correct / total if total else float("nan")

            # --- Validation: métricas + early stopping ---
            if has_validation and val_loader is not None:
                model.eval()
                val_epoch_loss = 0.0
                val_num_batches = 0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        logits = model(xb)
                        vloss = criterion(logits, yb)
                        val_epoch_loss += vloss.item()
                        val_num_batches += 1
                        vpreds = logits.argmax(dim=1)
                        val_correct += int(torch.eq(vpreds, yb).sum().item())
                        val_total += int(yb.numel())
                last_val_loss = val_epoch_loss / max(1, val_num_batches)
                last_val_acc = val_correct / val_total if val_total else float("nan")
                val_history.append(last_val_loss)

                if last_val_loss < best_val_loss:
                    best_val_loss = last_val_loss
                    best_state = copy.deepcopy(model.state_dict())
                    no_improve_streak = 0
                else:
                    no_improve_streak += 1
                    if no_improve_streak >= self._patience:
                        stopped_early = True

            logger.info(
                f"Epoch {epoch:02d}/{self._epochs} "
                f"train_loss={last_train_loss:.4f} train_acc={last_train_acc:.4f} "
                f"val_loss={last_val_loss:.4f} val_acc={last_val_acc:.4f}"
            )

            if stopped_early:
                break

        trained_epochs = epoch if self._epochs > 0 else 0

        # --- Checkpoint: restaurar el MEJOR estado (no el último epoch) ---
        best_epoch_reported, _ = _early_stopping_decision(val_history, self._patience)
        if has_validation and best_state is not None:
            model.load_state_dict(best_state)
            self._best_epoch = best_epoch_reported
            self._best_validation_loss = best_val_loss
            logger.info(
                f"{self.name}: checkpoint -> restaurado mejor estado "
                f"(best_epoch={self._best_epoch}, val_loss={best_val_loss:.4f})"
            )
        if stopped_early:
            logger.info(
                f"{self.name}: early stopping en epoch {trained_epochs} "
                f"(patience={self._patience}, best_epoch={self._best_epoch})"
            )

        self._is_trained = True
        return {
            "epochs": trained_epochs,
            "train_loss": last_train_loss,
            "train_accuracy": last_train_acc,
            "validation_loss": last_val_loss,
            "validation_accuracy": last_val_acc,
            "best_epoch": self._best_epoch,
            "best_validation_loss": (
                float("nan") if self._best_validation_loss is None else self._best_validation_loss
            ),
            "early_stopping": stopped_early,
            "patience": self._patience,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._prepare(X)
        model = self._ensure_model()
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X))
            return torch.softmax(logits, dim=1).numpy()

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
        }
        return metrics

    def save(self, path: str) -> None:
        model = self._ensure_model()
        torch.save(
            {"state_dict": model.state_dict(), "config": self._config()},
            path,
        )
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        state = torch.load(path, weights_only=False)
        self._apply_config(state["config"])
        self._model = _SequenceClassifier(self._build_network())
        self._model.load_state_dict(state["state_dict"])
        self._is_trained = True
        logger.info(f"Model loaded from {path}")

    def get_prediction(
        self,
        features: np.ndarray,
        symbol: str,
        timeframe: str,
        pattern_name: str,
    ) -> MLPrediction:
        if not self._is_trained:
            raise ValueError("Model is not trained yet")

        features = np.asarray(features, dtype=np.float32)
        if (
            self._scaler is not None
            and features.ndim == 2
            and features.shape[1] == self._feature_dim
        ):
            features = self._scaler.transform(features).astype(np.float32)
        if features.ndim == 1:
            if features.size == self._sequence_length * self._feature_dim:
                features = features.reshape(1, self._sequence_length, self._feature_dim)
            else:
                features = features.reshape(1, -1, 1)
        elif features.ndim == 2:
            if features.shape[1] == self._feature_dim:
                features = features.reshape(1, features.shape[0], self._feature_dim)
            elif features.size == self._sequence_length * self._feature_dim:
                features = features.reshape(1, self._sequence_length, self._feature_dim)
            else:
                features = features.reshape(features.shape[0], -1, self._feature_dim)

        probability = float(self.predict_proba(features)[0][1])
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
        probs = self.predict_proba(features)[0]
        margin = abs(float(probs[1]) - float(probs[0]))
        return float(min(1.0, 0.5 + margin))
