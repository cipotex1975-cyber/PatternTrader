from __future__ import annotations

from pydantic import BaseModel, Field

from app.market.fractals.models import Fractal
from app.market.pivots.models import Pivot
from app.market.trendlines.models import Channel, Trend, TrendDirection, Trendline
from app.market.zigzag.models import ZigZagPoint


class MarketStructure(BaseModel):
    """Complete price structure of a market, built by the MarketEngine."""

    symbol: str = ""
    timeframe: str = ""
    candle_count: int = 0
    indicators: dict[str, dict[str, float]] = Field(default_factory=dict)
    latest_indicators: dict[str, float] = Field(default_factory=dict)
    pivots: list[Pivot] = Field(default_factory=list)
    fractals: list[Fractal] = Field(default_factory=list)
    zigzag: list[ZigZagPoint] = Field(default_factory=list)
    trendlines: list[Trendline] = Field(default_factory=list)
    channels: list[Channel] = Field(default_factory=list)
    trend: Trend = Field(
        default_factory=lambda: Trend(direction=TrendDirection.SIDEWAYS, slope=0.0, strength=0.0)
    )

    @property
    def latest_pivot(self) -> Pivot | None:
        return self.pivots[-1] if self.pivots else None

    @property
    def has_channel(self) -> bool:
        return bool(self.channels)
