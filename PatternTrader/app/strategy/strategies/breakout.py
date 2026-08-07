from __future__ import annotations

from typing import Any

from app.core.logger import get_logger
from app.patterns.base_pattern import TradeDirection
from app.patterns.hypothesis import PatternHypothesis
from app.strategy.base import BaseStrategy, StrategyDecision
from app.strategy.registry import register_strategy
from app.strategy.utils import base_confidence, build_signal

logger = get_logger("BreakoutStrategy")

DEFAULT_PARAMETERS = {
    "min_score": 70.0,
    "min_health": 50.0,
    "default_size": 1.0,
    "min_momentum": 0.0,
    "rsi_min": 40.0,
    "rsi_max": 70.0,
}


@register_strategy
class BreakoutStrategy(BaseStrategy):
    """Entra en rupturas: momentum en la dirección y RSI en rango medio."""

    @property
    def name(self) -> str:
        return "breakout"

    @property
    def description(self) -> str:
        return "Entra en rupturas con momentum direccional y RSI en rango medio."

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
            return self._no_trade(
                f"Score {hypothesis.total_score:.1f} por debajo del mínimo"
            )
        if pattern.health < self._parameters["min_health"]:
            return self._no_trade(f"Salud {pattern.health:.1f} por debajo del mínimo")
        if (
            pattern.entry_price is None
            or pattern.stop_loss is None
            or pattern.take_profit is None
        ):
            return self._no_trade("Niveles de precio incompletos")

        rsi = indicators.get("rsi")
        momentum = indicators.get("momentum", 0.0)
        min_momentum = self._parameters["min_momentum"]

        if rsi is None:
            return self._no_trade("Indicador RSI no disponible")

        if pattern.direction == TradeDirection.LONG:
            if momentum < min_momentum:
                return self._no_trade(
                    f"Momentum {momentum:.2f} no soporta ruptura alcista"
                )
            if rsi < self._parameters["rsi_min"] or rsi > self._parameters["rsi_max"]:
                return self._no_trade(f"RSI {rsi:.1f} fuera de rango de ruptura")
            reasons = [
                f"Momentum alcista ({momentum:.2f})",
                f"RSI {rsi:.1f} en rango medio de ruptura",
            ]
        else:
            if momentum > -min_momentum:
                return self._no_trade(
                    f"Momentum {momentum:.2f} no soporta ruptura bajista"
                )
            short_rsi_min = 100.0 - self._parameters["rsi_max"]
            short_rsi_max = 100.0 - self._parameters["rsi_min"]
            if rsi < short_rsi_min or rsi > short_rsi_max:
                return self._no_trade(f"RSI {rsi:.1f} fuera de rango de ruptura")
            reasons = [
                f"Momentum bajista ({momentum:.2f})",
                f"RSI {rsi:.1f} en rango medio de ruptura",
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
