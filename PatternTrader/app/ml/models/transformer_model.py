from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from app.core.logger import get_logger
from app.ml.factory import MLModelFactory
from app.ml.models.sequence_base import SequenceModel

logger = get_logger("TransformerModel")


def _positional_encoding(seq_len: int, dim: int) -> torch.Tensor:
    pe = torch.zeros(seq_len, dim)
    position = torch.arange(seq_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe


class _TransformerNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int, nhead: int, num_layers: int) -> None:
        super().__init__()
        self.output_dim = hidden_dim
        self.embed = nn.Linear(input_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        h = self.embed(x) + _positional_encoding(seq_len, self.output_dim)
        out = self.encoder(h)
        return out.mean(dim=1)


class TransformerModel(SequenceModel):
    def __init__(self, nhead: int = 4, num_layers: int = 2, **kwargs: Any) -> None:
        self._nhead = nhead
        self._num_layers = num_layers
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "transformer"

    def _build_network(self) -> nn.Module:
        return _TransformerNetwork(
            self._feature_dim, self._hidden_dim, self._nhead, self._num_layers
        )

    def _config(self) -> dict[str, Any]:
        config = super()._config()
        config["nhead"] = self._nhead
        config["num_layers"] = self._num_layers
        return config

    def _apply_config(self, config: dict[str, Any]) -> None:
        super()._apply_config(config)
        self._nhead = config.get("nhead", 4)
        self._num_layers = config.get("num_layers", 2)


MLModelFactory.register("transformer", TransformerModel)
