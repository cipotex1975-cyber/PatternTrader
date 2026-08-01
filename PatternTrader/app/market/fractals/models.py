from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class FractalType(str, Enum):
    UP = "up"  # resistance / bearish fractal
    DOWN = "down"  # support / bullish fractal


class Fractal(BaseModel):
    index: int
    timestamp: datetime
    price: float
    type: FractalType

    model_config = {"frozen": True}
