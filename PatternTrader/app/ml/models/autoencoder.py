from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score

from app.core.logger import get_logger
from app.ml.base import BaseMLModel
from app.ml.factory import MLModelFactory

logger = get_logger("AutoEncoderModel")


class _AutoEncoderNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoEncoderModel(BaseMLModel):
    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 32,
        latent_dim: int = 8,
        epochs: int = 20,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._latent_dim = latent_dim
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._random_state = random_state
        self._model: nn.Module | None = None
        self._threshold: float = 0.0
        self._scale: float = 1.0

    @property
    def name(self) -> str:
        return "autoencoder"

    @property
    def model_type(self) -> str:
        return "anomaly"

    def _ensure_model(self) -> nn.Module:
        if self._model is None:
            torch.manual_seed(self._random_state)
            self._model = _AutoEncoderNetwork(self._input_dim, self._hidden_dim, self._latent_dim)
        return self._model

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return X.reshape(X.shape[0], -1)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        X = self._prepare(X)
        self._feature_names = feature_names or []

        model = self._ensure_model()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self._learning_rate)
        criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(torch.tensor(X))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        total_loss = 0.0
        for _ in range(self._epochs):
            for (xb,) in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), xb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        errors = self._reconstruction_errors(X)
        self._threshold = float(np.percentile(errors, 95))
        self._scale = float(errors.std()) if errors.std() > 0 else 1.0

        self._is_trained = True
        avg_loss = total_loss / max(1, len(loader))
        logger.info(
            f"Trained {self.name} ({self._epochs} epochs, loss {avg_loss:.4f}, "
            f"threshold {self._threshold:.4f})"
        )
        return {
            "epochs": self._epochs,
            "loss": avg_loss,
            "reconstruction_threshold": self._threshold,
        }

    def _reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        model = self._ensure_model()
        model.eval()
        with torch.no_grad():
            recon = model(torch.tensor(X))
            mse = ((torch.tensor(X) - recon) ** 2).mean(dim=1)
            return mse.numpy()

    def _anomaly_proba(self, X: np.ndarray) -> np.ndarray:
        errors = self._reconstruction_errors(self._prepare(X))
        return 1.0 / (1.0 + np.exp(-(errors - self._threshold) / self._scale))

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

        metrics: dict[str, float] = {"accuracy": accuracy_score(labels, predictions)}
        if len(np.unique(labels)) > 1:
            metrics["roc_auc"] = roc_auc_score(labels, probabilities)
        return metrics

    def _config(self) -> dict[str, Any]:
        return {
            "input_dim": self._input_dim,
            "hidden_dim": self._hidden_dim,
            "latent_dim": self._latent_dim,
            "epochs": self._epochs,
            "learning_rate": self._learning_rate,
            "batch_size": self._batch_size,
            "random_state": self._random_state,
            "threshold": self._threshold,
            "scale": self._scale,
        }

    def _apply_config(self, config: dict[str, Any]) -> None:
        self._input_dim = config.get("input_dim", self._input_dim)
        self._hidden_dim = config.get("hidden_dim", self._hidden_dim)
        self._latent_dim = config.get("latent_dim", self._latent_dim)
        self._epochs = config.get("epochs", self._epochs)
        self._learning_rate = config.get("learning_rate", self._learning_rate)
        self._batch_size = config.get("batch_size", self._batch_size)
        self._random_state = config.get("random_state", self._random_state)
        self._threshold = config.get("threshold", self._threshold)
        self._scale = config.get("scale", self._scale)

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
        self._model = _AutoEncoderNetwork(self._input_dim, self._hidden_dim, self._latent_dim)
        self._model.load_state_dict(state["state_dict"])
        self._is_trained = True
        logger.info(f"Model loaded from {path}")


MLModelFactory.register("autoencoder", AutoEncoderModel)
