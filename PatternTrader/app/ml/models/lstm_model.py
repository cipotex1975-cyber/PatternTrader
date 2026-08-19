from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from app.core.logger import get_logger
from app.ml.factory import MLModelFactory
from app.ml.models.sequence_base import SequenceModel

logger = get_logger("LSTMModel")


class _LSTMNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.output_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out[:, -1, :]


class LSTMModel(SequenceModel):
    def __init__(self, num_layers: int = 2, **kwargs: Any) -> None:
        self._num_layers = num_layers
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "lstm"

    def _build_network(self) -> nn.Module:
        return _LSTMNetwork(self._feature_dim, self._hidden_dim, self._num_layers)

    def _config(self) -> dict[str, Any]:
        config = super()._config()
        config["num_layers"] = self._num_layers
        return config

    def load(self, path: str) -> None:
        import torch

        from app.ml.models.sequence_base import _SequenceClassifier

        state = torch.load(path, weights_only=False)
        config = state["config"]
        self._num_layers = config.get("num_layers", 2)
        self._sequence_length = config.get("sequence_length", self._sequence_length)
        self._feature_dim = config.get("feature_dim", self._feature_dim)
        self._hidden_dim = config.get("hidden_dim", self._hidden_dim)
        self._epochs = config.get("epochs", self._epochs)
        self._learning_rate = config.get("learning_rate", self._learning_rate)
        self._batch_size = config.get("batch_size", self._batch_size)
        self._random_state = config.get("random_state", self._random_state)
        self._model = _SequenceClassifier(self._build_network())
        self._model.load_state_dict(state["state_dict"])
        self._is_trained = True
        logger.info(f"Model loaded from {path}")


MLModelFactory.register("lstm", LSTMModel)
