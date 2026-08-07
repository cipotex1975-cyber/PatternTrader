from __future__ import annotations

from typing import Any

from app.core.logger import get_logger
from app.patterns.base_pattern import PatternType, TradeDirection
from app.patterns.hypothesis import PatternHypothesis
from app.strategy.base import BaseStrategy, StrategyDecision
from app.strategy.registry import register_strategy
from app.strategy.utils import base_confidence, build_signal

logger = get_logger("ContrarianStrategy")

DEFAULT_PARAMETERS = {
    "min_score": 60.0,
    "min_health": 50.0,
    "default_size": 0.5,
    "oversold_rsi": 30.0,
    "overbought_rsi": 70.0,
    "max_reversal_momentum": 2.0,
}


@register_strategy
class ContrarianStrategy(BaseStrategy):
    """Entra en reversales contra la tendencia: RSI extremo + momentum que frena."""

    @property
    def name(self) -> str:
        return "contrarian"

    @property
    def description(self) -> str:
        return "Entra en reversales con RSI extremo y momentum perdiendo fuerza."

    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        self._parameters = {**DEFAULT_PARAMETERS, **(parameters or {})}

    def get_parameters(self) -> dict[str, Any]:
        return self._parameters.copy()

    def set_parameters(self, parameters: dict[str, Any]) -> None:
        self._parameters.update(parameters)

    def evaluate(self, hypothesis: PatternHypothesis) -> StrategyDecision:
        pattern = hypothesis.pattern
        indicators = hypothesis.indicators

        if pattern.pattern_type != PatternType.REVERSAL:
            return self._no_trade(
                f"Patrón {pattern.pattern_name} no es de reversa"
            )
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
        ema_fast = indicators.get("ema_9")
        ema_slow = indicators.get("ema_21")
        max_momentum = self._parameters["max_reversal_momentum"]

        if rsi is None or ema_fast is None or ema_slow is None:
            return self._no_trade("Indicadores necesarios no disponibles")

        if pattern.direction == TradeDirection.LONG:
            if rsi > self._parameters["oversold_rsi"]:
                return self._no_trade(f"RSI {rsi:.1f} no está sobrevendido")
            if momentum < -max_momentum:
                return self._no_trade(
                    f"Momentum {momentum:.2f} aún en caída fuerte"
                )
            if ema_fast >= ema_slow:
                return self._no_trade(
                    "Sin tendencia bajista previa (EMA9 >= EMA21)"
                )
            reasons = [
                f"RSI sobrevendido ({rsi:.1f})",
                f"Momentum estabilizando ({momentum:.2f})",
            ]
        else:
            if rsi < self._parameters["overbought_rsi"]:
                return self._no_trade(f"RSI {rsi:.1f} no está sobrecomprado")
            if momentum > max_momentum:
                return self._no_trade(
                    f"Momentum {momentum:.2f} aún en subida fuerte"
                )
            if ema_fast <= ema_slow:
                return self._no_trade(
                    "Sin tendencia alcista previa (EMA9 <= EMA21)"
                )
            reasons = [
                f"RSI sobrecomprado ({rsi:.1f})",
                f"Momentum perdiendo fuerza ({momentum:.2f})",
            ]

        signal = build_signal(
            hypothesis,
            self.name,
            size=self._parameters["default_size"],
            confidence=base_confidence(hypothesis) * 0.9,
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
