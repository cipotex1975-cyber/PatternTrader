from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class CandleData(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    model_config = {"frozen": True}

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def upper_shadow(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_shadow(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def is_doji(self) -> bool:
        return abs(self.close - self.open) < self.range * 0.1


class Candle(BaseModel):
    symbol: str
    timeframe: str
    data: CandleData
    indicators: dict[str, float] = Field(default_factory=dict)
    patterns: list[str] = Field(default_factory=list)
    metadata: dict[str, dict] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    def add_indicator(self, name: str, value: float) -> None:
        self.indicators[name] = value

    def add_pattern(self, pattern_name: str) -> None:
        if pattern_name not in self.patterns:
            self.patterns.append(pattern_name)

    def get_indicator(self, name: str) -> Optional[float]:
        return self.indicators.get(name)

    def has_pattern(self, pattern_name: str) -> bool:
        return pattern_name in self.patterns
