from __future__ import annotations

import numpy as np

from app.core.events.bus import get_event_bus
from app.core.events.models import Event, EventType
from app.core.logger import get_logger
from app.health.models import HealthFactor, HealthReport
from app.market.candles.models import Candle
from app.patterns.base_pattern import BasePattern, PatternResult, TradeDirection

logger = get_logger("HealthEngine")


class HealthEngine:
    """Recalcula la salud dinámica (0-100) de un patrón con cada vela.

    Factores evaluados: tiempo transcurrido, ATR, volumen, deformación,
    pendientes, rupturas falsas, volatilidad y tendencia principal.
    """

    FACTOR_WEIGHTS: dict[str, float] = {
        "time_decay": 0.20,
        "deformation": 0.20,
        "volume": 0.15,
        "trend": 0.10,
        "atr": 0.10,
        "slope": 0.10,
        "false_breakouts": 0.10,
        "volatility": 0.05,
    }

    def __init__(self) -> None:
        self._event_bus = get_event_bus()

    async def calculate(
        self,
        pattern: PatternResult,
        detector: BasePattern,
        candles: list[Candle],
        indicators: dict[str, float] | None = None,
    ) -> HealthReport:
        indicators = indicators or {}

        factors = [
            HealthFactor(
                name="time_decay",
                value=float(pattern.candles_remaining),
                score=self._score_time_decay(pattern),
                weight=self.FACTOR_WEIGHTS["time_decay"],
                reason="Velas restantes antes de expirar",
            ),
            HealthFactor(
                name="deformation",
                value=1.0 if detector.validate(pattern, candles) else 0.0,
                score=self._score_deformation(detector, pattern, candles),
                weight=self.FACTOR_WEIGHTS["deformation"],
                reason="Integridad de la estructura del patrón",
            ),
            HealthFactor(
                name="volume",
                value=indicators.get("volume", 0),
                score=self._score_volume(indicators, candles),
                weight=self.FACTOR_WEIGHTS["volume"],
                reason="Volumen confirma el movimiento",
            ),
            HealthFactor(
                name="trend",
                value=indicators.get("ema_21", 0) - indicators.get("ema_50", 0),
                score=self._score_trend(indicators, pattern),
                weight=self.FACTOR_WEIGHTS["trend"],
                reason="Alineación con la tendencia principal",
            ),
            HealthFactor(
                name="atr",
                value=indicators.get("atr", 0),
                score=self._score_atr(indicators, candles),
                weight=self.FACTOR_WEIGHTS["atr"],
                reason="Volatilidad ATR suficiente",
            ),
            HealthFactor(
                name="slope",
                value=0.0,
                score=self._score_slope(candles, pattern),
                weight=self.FACTOR_WEIGHTS["slope"],
                reason="Pendiente reciente del precio",
            ),
            HealthFactor(
                name="false_breakouts",
                value=0.0,
                score=self._score_false_breakouts(pattern, candles),
                weight=self.FACTOR_WEIGHTS["false_breakouts"],
                reason="Rupturas falsas del nivel clave",
            ),
            HealthFactor(
                name="volatility",
                value=indicators.get("atr", 0),
                score=self._score_volatility(indicators, candles),
                weight=self.FACTOR_WEIGHTS["volatility"],
                reason="Consistencia de la volatilidad",
            ),
        ]

        total_weight = sum(f.weight for f in factors)
        health = sum(f.score * f.weight for f in factors) / total_weight
        health = max(0.0, min(100.0, health))

        report = HealthReport(
            health=health,
            factors=factors,
            metadata={
                "pattern_id": str(pattern.id),
                "pattern_name": pattern.pattern_name,
                "symbol": pattern.symbol,
                "timeframe": pattern.timeframe,
            },
        )

        await self._event_bus.publish(
            Event(
                type=EventType.HEALTH_UPDATED,
                source="HealthEngine",
                data={
                    "pattern_id": str(pattern.id),
                    "health": report.health,
                    "weakest": report.weakest_factor.name if report.weakest_factor else None,
                },
            )
        )

        logger.debug(f"Health for {pattern.pattern_name} {pattern.symbol}: {report.health:.1f}")
        return report

    def _score_time_decay(self, pattern: PatternResult) -> float:
        total = pattern.max_confirmation_candles
        if total <= 0:
            return 100.0
        return 100.0 * (pattern.candles_remaining / total)

    def _score_deformation(
        self, detector: BasePattern, pattern: PatternResult, candles: list[Candle]
    ) -> float:
        try:
            valid = detector.validate(pattern, candles)
        except Exception:
            valid = True
        return 90.0 if valid else 30.0

    def _score_volume(self, indicators: dict[str, float], candles: list[Candle]) -> float:
        if not candles or len(candles) < 20:
            return 50.0

        recent_volume = sum(c.data.volume for c in candles[-5:]) / 5
        avg_volume = sum(c.data.volume for c in candles[-20:]) / 20

        if avg_volume == 0:
            return 50.0

        ratio = recent_volume / avg_volume
        if ratio > 1.5:
            return 90.0
        elif ratio > 1.2:
            return 75.0
        elif ratio > 0.8:
            return 60.0
        else:
            return 40.0

    def _score_trend(self, indicators: dict[str, float], pattern: PatternResult) -> float:
        ema_21 = indicators.get("ema_21", 0)
        ema_50 = indicators.get("ema_50", 0)
        if ema_21 == 0 or ema_50 == 0:
            return 50.0

        if pattern.direction == TradeDirection.LONG:
            aligned = ema_21 > ema_50
        else:
            aligned = ema_21 < ema_50
        return 90.0 if aligned else 45.0

    def _score_atr(self, indicators: dict[str, float], candles: list[Candle]) -> float:
        if not candles or len(candles) < 20:
            return 50.0

        current_atr = indicators.get("atr", 0)
        if current_atr == 0:
            return 50.0

        avg_price = sum(c.data.close for c in candles[-20:]) / 20
        if avg_price == 0:
            return 50.0

        atr_pct = (current_atr / avg_price) * 100
        if 0.5 < atr_pct < 2.0:
            return 80.0
        elif 0.3 < atr_pct < 3.0:
            return 60.0
        else:
            return 40.0

    def _score_slope(self, candles: list[Candle], pattern: PatternResult) -> float:
        if not candles or len(candles) < 5:
            return 60.0

        closes = np.array([c.data.close for c in candles[-20:]])
        x = np.arange(len(closes))
        slope = float(np.polyfit(x, closes, 1)[0])

        mean_price = float(closes.mean())
        if mean_price == 0:
            return 60.0

        relative_slope = slope / mean_price
        max_relative = 0.005
        normalized = float(np.clip(relative_slope / max_relative, -1.0, 1.0))

        if pattern.direction == TradeDirection.LONG:
            return 50.0 + 50.0 * max(0.0, normalized)
        return 50.0 + 50.0 * max(0.0, -normalized)

    def _score_false_breakouts(self, pattern: PatternResult, candles: list[Candle]) -> float:
        neckline = pattern.key_levels.get("neckline", 0)
        if neckline == 0 or not candles or len(candles) < 5:
            return 100.0

        if pattern.direction == TradeDirection.LONG:
            count = sum(
                1 for c in candles[-10:] if c.data.high > neckline and c.data.close <= neckline
            )
        else:
            count = sum(
                1 for c in candles[-10:] if c.data.low < neckline and c.data.close >= neckline
            )

        return max(0.0, 100.0 - count * 25.0)

    def _score_volatility(self, indicators: dict[str, float], candles: list[Candle]) -> float:
        if not candles or len(candles) < 20:
            return 60.0

        ranges = np.array([c.data.range for c in candles[-20:]])
        atr = indicators.get("atr", float(np.mean(ranges)))
        if atr == 0:
            return 60.0

        ratio = float(np.mean(ranges) / atr)
        if 0.7 <= ratio <= 1.3:
            return 85.0
        elif 0.5 <= ratio <= 1.5:
            return 65.0
        else:
            return 40.0
