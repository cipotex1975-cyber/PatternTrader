from __future__ import annotations

from typing import Any, Optional

from app.core.config.settings import get_settings
from app.core.logger import get_logger
from app.patterns.hypothesis import PatternHypothesis
from app.strategy.engine import StrategyEngine, StrategyEngineResult
from app.strategy.registry import StrategyRegistry

logger = get_logger("StrategyManager")


class StrategyManager:
    """Gestión en runtime de las estrategias registradas.

    Construye las estrategias habilitadas vía ``StrategyFactory`` (desde
    ``settings.strategies``) y permite activar/desactivar o ajustar parámetros
    en caliente. ``evaluate`` delega en el ``StrategyEngine`` actualizado.
    """

    def __init__(
        self,
        enabled: Optional[list[str]] = None,
        params: Optional[dict[str, dict[str, Any]]] = None,
    ) -> None:
        settings = get_settings()
        enabled_defaults = settings.strategies.enabled
        params_defaults = settings.strategies.params

        self._enabled = {
            name: name in (enabled if enabled is not None else enabled_defaults)
            for name in StrategyRegistry.get_all()
        }
        self._params = {
            name: dict(parameters)
            for name, parameters in (params if params is not None else params_defaults).items()
        }
        self._engine = self._build_engine()

    @property
    def engine(self) -> StrategyEngine:
        return self._engine

    def _build_engine(self) -> StrategyEngine:
        enabled = [name for name, flag in self._enabled.items() if flag]
        return StrategyEngine(strategies=enabled, parameters=self._params)

    def evaluate(self, hypothesis: PatternHypothesis) -> StrategyEngineResult:
        return self._engine.evaluate(hypothesis)

    def list(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for name, strategy_class in StrategyRegistry.get_all().items():
            instance = strategy_class()
            result.append(
                {
                    "name": name,
                    "description": instance.description,
                    "version": instance.version,
                    "enabled": self._enabled.get(name, False),
                    "parameters": self._params.get(name) or instance.get_parameters(),
                }
            )
        return sorted(result, key=lambda item: item["name"])

    def enable(self, name: str) -> bool:
        if not self._is_registered(name):
            return False
        if not self._enabled.get(name, False):
            self._enabled[name] = True
            self._rebuild()
            logger.info(f"Strategy enabled: {name}")
        return True

    def disable(self, name: str) -> bool:
        if not self._is_registered(name):
            return False
        if self._enabled.get(name, False):
            self._enabled[name] = False
            self._rebuild()
            logger.info(f"Strategy disabled: {name}")
        return True

    def set_params(self, name: str, parameters: dict[str, Any]) -> bool:
        if not self._is_registered(name):
            return False
        self._params[name] = {**(self._params.get(name) or {}), **parameters}
        self._rebuild()
        logger.info(f"Strategy params updated: {name} {parameters}")
        return True

    def reset(self) -> None:
        settings = get_settings()
        self._enabled = {
            name: name in settings.strategies.enabled for name in StrategyRegistry.get_all()
        }
        self._params = {name: dict(p) for name, p in settings.strategies.params.items()}
        self._rebuild()
        logger.info("StrategyManager reset to settings defaults")

    def _rebuild(self) -> None:
        self._engine = self._build_engine()

    def _is_registered(self, name: str) -> bool:
        return name in StrategyRegistry.get_all()
