from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class PivotType(str, Enum):
    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"


class Pivot(BaseModel):
    index: int
    timestamp: datetime
    price: float
    type: PivotType

    model_config = {"frozen": True}
