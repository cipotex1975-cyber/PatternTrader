from __future__ import annotations

from typing import Any

from app.core.exceptions import ConfigurationError
from app.core.logger import get_logger
from app.ml.base import BaseMLModel

logger = get_logger("MLModelFactory")


class MLModelFactory:
    _models: dict[str, type[BaseMLModel]] = {}
    _instances: dict[str, BaseMLModel] = {}

    @classmethod
    def register(cls, name: str, model_class: type[BaseMLModel]) -> None:
        cls._models[name] = model_class
        logger.debug(f"Registered ML model: {name}")

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseMLModel:
        if name not in cls._models:
            raise ConfigurationError(f"Unknown ML model: {name}")

        if name not in cls._instances:
            cls._instances[name] = cls._models[name](**kwargs)

        return cls._instances[name]

    @classmethod
    def get_all(cls) -> dict[str, type[BaseMLModel]]:
        return cls._models.copy()

    @classmethod
    def clear(cls) -> None:
        cls._instances.clear()
