from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class ZigZagType(str, Enum):
    HIGH = "high"
    LOW = "low"


class ZigZagPoint(BaseModel):
    index: int
    timestamp: datetime
    price: float
    type: ZigZagType

    model_config = {"frozen": True}
