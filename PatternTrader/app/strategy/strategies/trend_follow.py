from __future__ import annotations

from typing import Any

from app.core.logger import get_logger
from app.patterns.base_pattern import TradeDirection
from app.patterns.hypothesis import PatternHypothesis
from app.strategy.base import BaseStrategy, StrategyDecision
from app.strategy.registry import register_strategy
from app.strategy.utils import base_confidence, build_signal

logger = get_logger("TrendFollowStrategy")

DEFAULT_PARAMETERS = {
    "min_score": 70.0,
    "min_health": 55.0,
    "default_size": 1.0,
    "momentum_threshold": 0.0,
}


@register_strategy
class TrendFollowStrategy(BaseStrategy):
    """Entra a favor de la tendencia dominante (EMA9 vs EMA21 + momentum)."""

    @property
    def name(self) -> str:
        return "trend_follow"

    @property
    def description(self) -> str:
        return "Entra a favor de la tendencia (EMA9 vs EMA21 + momentum)."

    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        self._parameters = {**DEFAULT_PARAMETERS, **(parameters or {})}

    def get_parameters(self) -> dict[str, Any]:
        return self._parameters.copy()

    def set_parameters(self, parameters: dict[str, Any]) -> None:
        self._parameters.update(parameters)

    def evaluate(self, hypothesis: PatternHypothesis) -> StrategyDecision:
        pattern = hypothesis.pattern
        indicators = hypothesis.indicators

        if hypothesis.total_score < self._parameters["min_score"]:
            return self._no_trade(f"Score {hypothesis.total_score:.1f} por debajo del mínimo")
        if pattern.health < self._parameters["min_health"]:
            return self._no_trade(f"Salud {pattern.health:.1f} por debajo del mínimo")
        if pattern.entry_price is None or pattern.stop_loss is None or pattern.take_profit is None:
            return self._no_trade("Niveles de precio incompletos")

        ema_fast = indicators.get("ema_9")
        ema_slow = indicators.get("ema_21")
        momentum = indicators.get("momentum", 0.0)
        threshold = self._parameters["momentum_threshold"]

        if ema_fast is None or ema_slow is None:
            return self._no_trade("Indicadores EMA incompletos")

        if pattern.direction == TradeDirection.LONG:
            if ema_fast <= ema_slow:
                return self._no_trade("Tendencia no alcista (EMA9 <= EMA21)")
            if momentum < threshold:
                return self._no_trade(f"Momentum {momentum:.2f} no soporta entrada larga")
            reasons = [
                "Tendencia alcista (EMA9 > EMA21)",
                f"Momentum positivo ({momentum:.2f})",
            ]
        else:
            if ema_fast >= ema_slow:
                return self._no_trade("Tendencia no bajista (EMA9 >= EMA21)")
            if momentum > -threshold:
                return self._no_trade(f"Momentum {momentum:.2f} no soporta entrada corta")
            reasons = [
                "Tendencia bajista (EMA9 < EMA21)",
                f"Momentum negativo ({momentum:.2f})",
            ]

        signal = build_signal(
            hypothesis,
            self.name,
            size=self._parameters["default_size"],
            confidence=base_confidence(hypothesis),
            reasons=reasons,
            metadata={"strategy": self.name, "parameters": self.get_parameters()},
        )
        if signal is None:
            return self._no_trade("No se pudo construir la señal")

        return StrategyDecision(
            strategy_name=self.name,
            action="ENTER",
            signal=signal,
            reasons=reasons,
            confidence=signal.confidence,
        )
