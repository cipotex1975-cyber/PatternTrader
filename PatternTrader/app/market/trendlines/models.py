from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class TrendDirection(str, Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"


class TrendlineType(str, Enum):
    SUPPORT = "support"
    RESISTANCE = "resistance"


class Trendline(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    type: TrendlineType
    start_index: int
    end_index: int
    start_price: float
    end_price: float
    start_timestamp: datetime
    end_timestamp: datetime
    slope: float
    angle_degrees: float
    touches: int = 0


class Trend(BaseModel):
    direction: TrendDirection
    slope: float
    strength: float = Field(ge=0.0, le=100.0)
    higher_highs: int = 0
    higher_lows: int = 0
    lower_highs: int = 0
    lower_lows: int = 0


class Channel(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    direction: TrendDirection
    upper: Trendline
    lower: Trendline
    slope: float
    width: float
    touches_upper: int
    touches_lower: int
