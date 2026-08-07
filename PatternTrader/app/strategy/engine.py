from __future__ import annotations

from typing import Any, Optional

from app.core.config.settings import get_settings
from app.core.logger import get_logger
from app.patterns.hypothesis import PatternHypothesis
from app.strategy.base import BaseStrategy, StrategyDecision
from app.strategy.registry import StrategyRegistry

logger = get_logger("StrategyEngine")


class StrategyEngineResult:
    def __init__(self, decisions: list[StrategyDecision], best: Optional[StrategyDecision]) -> None:
        self.decisions = decisions
        self.best = best

    @property
    def has_entry(self) -> bool:
        return self.best is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions": [decision.to_dict() for decision in self.decisions],
            "best": self.best.to_dict() if self.best else None,
        }


class StrategyEngine:
    """Ejecuta las estrategias registradas sobre una hipótesis y elige la mejor.

    Las estrategias son el paso 2 del pipeline: deciden si la hipótesis del
    patrón merece una entrada.
    """

    def __init__(
        self,
        strategies: Optional[list[str]] = None,
        parameters: Optional[dict[str, dict[str, Any]]] = None,
    ) -> None:
        settings = get_settings()
        enabled = strategies if strategies is not None else settings.strategies.enabled
        params = parameters if parameters is not None else settings.strategies.params

        self._strategies: list[BaseStrategy] = []
        for name in enabled:
            strategy_class = StrategyRegistry.get(name)
            if strategy_class is None:
                logger.warning(f"Estrategia '{name}' no registrada; se ignora")
                continue
            strategy = strategy_class()
            strategy.set_parameters(params.get(name) or {})
            self._strategies.append(strategy)

    @property
    def strategies(self) -> list[BaseStrategy]:
        return list(self._strategies)

    @property
    def strategy_names(self) -> list[str]:
        return [strategy.name for strategy in self._strategies]

    def evaluate(self, hypothesis: PatternHypothesis) -> StrategyEngineResult:
        decisions: list[StrategyDecision] = []
        for strategy in self._strategies:
            try:
                decisions.append(strategy.evaluate(hypothesis))
            except Exception as exc:  # pragma: no cover - protección defensiva
                logger.exception(f"Estrategia {strategy.name} falló: {exc}")

        best: Optional[StrategyDecision] = None
        for decision in decisions:
            if decision.is_entry and (
                best is None or decision.confidence > best.confidence
            ):
                best = decision

        return StrategyEngineResult(decisions=decisions, best=best)
