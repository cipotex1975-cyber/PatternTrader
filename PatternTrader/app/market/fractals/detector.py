from __future__ import annotations

from app.core.config.settings import get_settings
from app.market.candles.models import Candle
from app.market.fractals.models import Fractal, FractalType


class FractalDetector:
    """Detects Bill Williams fractals in a candle series.

    An up fractal is a bar whose high is strictly higher than the highs of
    ``window`` bars on each side; a down fractal is a bar whose low is
    strictly lower than the lows of ``window`` bars on each side.
    """

    def __init__(self, window: int | None = None) -> None:
        if window is None:
            settings = get_settings()
            window = settings.market.structure.fractal_window
        self._window = max(1, window)

    @property
    def window(self) -> int:
        return self._window

    def detect(self, candles: list[Candle]) -> list[Fractal]:
        n = len(candles)
        window = self._window
        if n < (2 * window) + 1:
            return []

        fractals: list[Fractal] = []
        for i in range(window, n - window):
            high = candles[i].data.high
            low = candles[i].data.low

            is_up = all(
                candles[j].data.high < high
                for j in list(range(i - window, i)) + list(range(i + 1, i + window + 1))
            )
            is_down = all(
                candles[j].data.low > low
                for j in list(range(i - window, i)) + list(range(i + 1, i + window + 1))
            )

            if is_up:
                fractals.append(
                    Fractal(
                        index=i,
                        timestamp=candles[i].data.timestamp,
                        price=float(high),
                        type=FractalType.UP,
                    )
                )
            elif is_down:
                fractals.append(
                    Fractal(
                        index=i,
                        timestamp=candles[i].data.timestamp,
                        price=float(low),
                        type=FractalType.DOWN,
                    )
                )

        return fractals

    def detect_up(self, candles: list[Candle]) -> list[Fractal]:
        return [f for f in self.detect(candles) if f.type == FractalType.UP]

    def detect_down(self, candles: list[Candle]) -> list[Fractal]:
        return [f for f in self.detect(candles) if f.type == FractalType.DOWN]
