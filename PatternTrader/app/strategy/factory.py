from __future__ import annotations

from typing import Any, Optional

from app.core.logger import get_logger
from app.strategy.base import BaseStrategy
from app.strategy.registry import StrategyRegistry

logger = get_logger("StrategyFactory")


class StrategyFactory:
    """Factory para crear instancias de estrategias con parámetros."""

    @classmethod
    def create(cls, name: str, parameters: Optional[dict[str, Any]] = None) -> BaseStrategy:
        strategy_class = StrategyRegistry.get(name)
        if strategy_class is None:
            raise ValueError(f"Unknown strategy: {name}")
        strategy = strategy_class()
        if parameters:
            strategy.set_parameters(parameters)
        return strategy

    @classmethod
    def create_all(
        cls, parameters: Optional[dict[str, dict[str, Any]]] = None
    ) -> list[BaseStrategy]:
        parameters = parameters or {}
        return [
            cls.create(name, parameters.get(name))
            for name in StrategyRegistry.get_all()
        ]
