from __future__ import annotations

from typing import Type

from app.core.logger import get_logger
from app.strategy.base import BaseStrategy

logger = get_logger("StrategyRegistry")


class StrategyRegistry:
    _strategies: dict[str, Type[BaseStrategy]] = {}

    @classmethod
    def register(cls, strategy_class: Type[BaseStrategy]) -> None:
        instance = strategy_class()
        cls._strategies[instance.name] = strategy_class
        logger.debug(f"Registered strategy: {instance.name}")

    @classmethod
    def get(cls, name: str) -> Type[BaseStrategy] | None:
        return cls._strategies.get(name)

    @classmethod
    def get_all(cls) -> dict[str, Type[BaseStrategy]]:
        return cls._strategies.copy()

    @classmethod
    def get_all_instances(cls) -> list[BaseStrategy]:
        return [strategy_class() for strategy_class in cls._strategies.values()]

    @classmethod
    def clear(cls) -> None:
        cls._strategies.clear()


def register_strategy(strategy_class: Type[BaseStrategy]) -> Type[BaseStrategy]:
    StrategyRegistry.register(strategy_class)
    return strategy_class
