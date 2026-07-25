from __future__ import annotations

from typing import Type

from app.core.logger import get_logger
from app.patterns.base_pattern import BasePattern

logger = get_logger("PatternRegistry")


class PatternRegistry:
    _patterns: dict[str, Type[BasePattern]] = {}

    @classmethod
    def register(cls, pattern_class: Type[BasePattern]) -> None:
        instance = pattern_class()
        cls._patterns[instance.name] = pattern_class
        logger.debug(f"Registered pattern: {instance.name}")

    @classmethod
    def get(cls, name: str) -> Type[BasePattern] | None:
        return cls._patterns.get(name)

    @classmethod
    def get_all(cls) -> dict[str, Type[BasePattern]]:
        return cls._patterns.copy()

    @classmethod
    def get_all_instances(cls) -> list[BasePattern]:
        return [pattern_class() for pattern_class in cls._patterns.values()]

    @classmethod
    def clear(cls) -> None:
        cls._patterns.clear()


def register_pattern(pattern_class: Type[BasePattern]) -> Type[BasePattern]:
    PatternRegistry.register(pattern_class)
    return pattern_class
