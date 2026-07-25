from __future__ import annotations

from typing import Any

from app.core.config.settings import get_settings
from app.core.exceptions import ConfigurationError
from app.core.logger import get_logger
from app.data.providers.base import IDataProvider

logger = get_logger("DataProviderFactory")


class DataProviderFactory:
    _providers: dict[str, type[IDataProvider]] = {}
    _instances: dict[str, IDataProvider] = {}

    @classmethod
    def register(cls, name: str, provider_class: type[IDataProvider]) -> None:
        cls._providers[name] = provider_class
        logger.debug(f"Registered data provider: {name}")

    @classmethod
    def create(cls, name: str | None = None, **kwargs: Any) -> IDataProvider:
        if name is None:
            settings = get_settings()
            name = settings.data_providers.default

        if name not in cls._providers:
            raise ConfigurationError(f"Unknown data provider: {name}")

        if name not in cls._instances:
            cls._instances[name] = cls._providers[name](**kwargs)

        return cls._instances[name]

    @classmethod
    def get_all(cls) -> dict[str, type[IDataProvider]]:
        return cls._providers.copy()

    @classmethod
    def clear(cls) -> None:
        cls._instances.clear()
