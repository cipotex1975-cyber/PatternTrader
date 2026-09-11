from __future__ import annotations

from typing import Optional

from app.core.config.settings import get_settings
from app.core.logger import get_logger
from app.data.providers.base import IDataProvider
from app.data.providers.factory import DataProviderFactory
from app.database.repositories import LifecycleRepository, SignalRepository, TradeRepository
from app.execution.engine import ExecutionEngine
from app.patterns.pipeline import PatternPipeline
from app.risk.engine import RiskEngine
from app.scheduler.main import Scheduler
from app.strategy.manager import StrategyManager

logger = get_logger("PatternService")


def resolve_provider_names(
    symbols: list[str],
    symbol_providers: dict[str, str],
    default_provider: str,
) -> list[str]:
    """Devuelve la lista ordenada y sin duplicados de proveedores necesarios."""
    names: list[str] = []
    for symbol in symbols:
        name = symbol_providers.get(symbol, default_provider)
        if name not in names:
            names.append(name)
    return names


def resolve_symbol_provider(
    symbol: str,
    symbol_providers: dict[str, str],
    default_provider: str,
) -> str:
    return symbol_providers.get(symbol, default_provider)


class PatternService:
    """Ejecuta el pipeline de patrones de forma periódica para cada símbolo/timeframe."""

    def __init__(
        self,
        learning_service: Optional[object] = None,
        lifecycle_repository: Optional[object] = None,
        signal_repository: Optional[object] = None,
        trade_repository: Optional[object] = None,
    ) -> None:
        settings = get_settings()
        self._settings = settings
        lifecycle_settings = settings.patterns.lifecycle
        self._enabled = lifecycle_settings.enabled
        self._interval_seconds = lifecycle_settings.check_interval_seconds
        self._symbols = settings.market.default_symbols
        self._timeframes = lifecycle_settings.timeframes
        self._candle_limit = lifecycle_settings.candle_limit

        self._providers: dict[str, IDataProvider] = {}
        self._provider_route: dict[str, str] = {}
        self._risk = RiskEngine(
            symbol_sectors=settings.risk.symbol_sectors,
            correlations=settings.risk.correlations,
        )
        self._strategy_manager = StrategyManager()
        self._pipeline = PatternPipeline(
            max_candles=self._candle_limit,
            learning_service=learning_service,
            lifecycle_repository=lifecycle_repository or LifecycleRepository(),
            signal_repository=signal_repository or SignalRepository(),
            risk_engine=self._risk,
            strategy_manager=self._strategy_manager,
        )
        self._execution = ExecutionEngine(
            lifecycle=self._pipeline.lifecycle,
            repository=trade_repository or TradeRepository(),
            risk_engine=self._risk,
        )
        self._scheduler = Scheduler()

    @property
    def pipeline(self) -> PatternPipeline:
        return self._pipeline

    @property
    def execution(self) -> ExecutionEngine:
        return self._execution

    @property
    def strategy_manager(self) -> StrategyManager:
        return self._strategy_manager

    async def start(self) -> None:
        if not self._enabled:
            logger.info("PatternService disabled; pipeline not started")
            return

        await self._execution.start()

        await self._connect_providers()

        await self._rehydrate_lifecycle()
        await self._scheduler.start()

        for symbol in self._symbols:
            for timeframe in self._timeframes:
                await self._scheduler.add_interval(
                    name=f"pattern_pipeline_{symbol}_{timeframe}",
                    func=self._pipeline.process_symbol,
                    interval_seconds=self._interval_seconds,
                    symbol=symbol,
                    timeframe=timeframe,
                )

        logger.info(
            f"PatternService started: {len(self._symbols)} symbols x "
            f"{len(self._timeframes)} timeframes"
        )

    async def stop(self) -> None:
        await self._scheduler.stop()
        await self._execution.stop()
        for name, provider in list(self._providers.items()):
            try:
                await provider.disconnect()
            except Exception as e:
                logger.error(f"Failed to disconnect provider '{name}': {e}")
        self._providers.clear()
        self._provider_route.clear()
        logger.info("PatternService stopped")

    def get_scheduler_tasks(self) -> list[str]:
        return self._scheduler.get_tasks()

    def _resolver(self, symbol: str, timeframe: str) -> IDataProvider | None:
        name = self._provider_route.get(symbol)
        if name is None:
            return None
        return self._providers.get(name)

    async def _connect_providers(self) -> None:
        symbol_providers = self._settings.market.symbol_providers
        default_provider = self._settings.data_providers.default

        names = resolve_provider_names(self._symbols, symbol_providers, default_provider)
        self._provider_route = {
            symbol: resolve_symbol_provider(symbol, symbol_providers, default_provider)
            for symbol in self._symbols
        }

        for name in names:
            try:
                provider = DataProviderFactory.create(name)
                await provider.connect()
                self._providers[name] = provider
                logger.info(f"Data provider connected: {name}")
            except Exception as e:
                logger.error(f"Failed to connect data provider '{name}': {e}")
                self._provider_route = {
                    symbol: provider_name
                    for symbol, provider_name in self._provider_route.items()
                    if provider_name != name
                }

        if self._providers:
            self._pipeline.set_provider_resolver(self._resolver)

    async def _rehydrate_lifecycle(self) -> None:
        await self._pipeline.lifecycle.rehydrate_from_db()
