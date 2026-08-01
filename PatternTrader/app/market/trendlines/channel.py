from __future__ import annotations

from app.core.config.settings import get_settings
from app.market.candles.models import Candle
from app.market.trendlines.detector import TrendlineDetector
from app.market.trendlines.models import Channel, TrendDirection, Trendline, TrendlineType


class ChannelDetector:
    """Detects price channels from parallel support/resistance trendlines.

    A channel is a pair of trendlines (one support, one resistance) with a
    similar slope whose index ranges overlap.
    """

    def __init__(self, slope_tolerance: float | None = None) -> None:
        settings = get_settings()
        if slope_tolerance is None:
            slope_tolerance = settings.market.structure.channel_slope_tolerance
        self._slope_tolerance = slope_tolerance

    @property
    def slope_tolerance(self) -> float:
        return self._slope_tolerance

    def detect(
        self,
        candles: list[Candle],
        trendlines: list[Trendline] | None = None,
    ) -> list[Channel]:
        if trendlines is None:
            trendlines = TrendlineDetector().detect(candles)

        supports = [t for t in trendlines if t.type == TrendlineType.SUPPORT]
        resistances = [t for t in trendlines if t.type == TrendlineType.RESISTANCE]

        channels: list[Channel] = []
        for resistance in resistances:
            for support in supports:
                if not self._overlap(resistance, support):
                    continue
                if not self._parallel(resistance.slope, support.slope):
                    continue

                direction = self._direction(resistance.slope, support.slope)
                start = max(resistance.start_index, support.start_index)
                end = min(resistance.end_index, support.end_index)

                detector = TrendlineDetector()
                upper_at_start = detector.line_value_at(resistance, start)
                lower_at_start = detector.line_value_at(support, start)
                upper_at_end = detector.line_value_at(resistance, end)
                lower_at_end = detector.line_value_at(support, end)
                width = ((upper_at_start - lower_at_start) + (upper_at_end - lower_at_end)) / 2.0
                if width <= 0:
                    continue

                channels.append(
                    Channel(
                        direction=direction,
                        upper=resistance,
                        lower=support,
                        slope=resistance.slope,
                        width=round(abs(width), 8),
                        touches_upper=resistance.touches,
                        touches_lower=support.touches,
                    )
                )

        return channels

    @staticmethod
    def _overlap(a: Trendline, b: Trendline) -> bool:
        return a.start_index <= b.end_index and b.start_index <= a.end_index

    def _parallel(self, slope_a: float, slope_b: float) -> bool:
        reference = max(abs(slope_a), abs(slope_b), 1e-9)
        return abs(slope_a - slope_b) / reference <= self._slope_tolerance

    @staticmethod
    def _direction(slope_a: float, slope_b: float) -> TrendDirection:
        avg = (slope_a + slope_b) / 2.0
        if avg > 0:
            return TrendDirection.UPTREND
        if avg < 0:
            return TrendDirection.DOWNTREND
        return TrendDirection.SIDEWAYS
