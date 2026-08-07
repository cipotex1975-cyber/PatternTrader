from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from app.core.logger import get_logger
from app.ml.factory import MLModelFactory
from app.ml.models.sequence_base import SequenceModel

logger = get_logger("CNNModel")


class _CNNNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int, kernel_size: int) -> None:
        super().__init__()
        self.output_dim = hidden_dim
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(input_size, hidden_dim, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        h = self.relu(self.conv1(h))
        h = self.relu(self.conv2(h))
        h = self.pool(h).squeeze(-1)
        return h


class CNNModel(SequenceModel):
    def __init__(self, kernel_size: int = 3, **kwargs: Any) -> None:
        self._kernel_size = kernel_size
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "cnn"

    def _build_network(self) -> nn.Module:
        return _CNNNetwork(self._feature_dim, self._hidden_dim, self._kernel_size)

    def _config(self) -> dict[str, Any]:
        config = super()._config()
        config["kernel_size"] = self._kernel_size
        return config

    def _apply_config(self, config: dict[str, Any]) -> None:
        super()._apply_config(config)
        self._kernel_size = config.get("kernel_size", 3)


MLModelFactory.register("cnn", CNNModel)
