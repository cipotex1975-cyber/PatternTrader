from __future__ import annotations

from typing import Optional

from app.core.config.settings import get_settings
from app.core.logger import get_logger
from app.data.providers.base import IDataProvider
from app.data.providers.factory import DataProviderFactory
from app.database.repositories import LifecycleRepository, SignalRepository, TradeRepository
from app.execution.engine import ExecutionEngine
from app.patterns.pipeline import PatternPipeline, timeframe_to_seconds
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
        signal_repository: Optional[SignalRepository] = None,
        trade_repository: Optional[object] = None,
    ) -> None:
        settings = get_settings()
        self._settings = settings
        lifecycle_settings = settings.patterns.lifecycle
        self._enabled = lifecycle_settings.enabled
        self._interval_seconds = lifecycle_settings.check_interval_seconds
        self._checks_per_candle = lifecycle_settings.polling_checks_per_candle
        self._symbols = settings.market.default_symbols
        self._timeframes = lifecycle_settings.timeframes
        self._candle_limit = lifecycle_settings.candle_limit

        self._providers: dict[str, IDataProvider] = {}
        self._provider_route: dict[str, str] = {}
        self._signal_repository: SignalRepository = signal_repository or SignalRepository()
        self._risk = RiskEngine(
            symbol_sectors=settings.risk.symbol_sectors,
            correlations=settings.risk.correlations,
        )
        self._strategy_manager = StrategyManager()
        self._pipeline = PatternPipeline(
            max_candles=self._candle_limit,
            learning_service=learning_service,
            lifecycle_repository=lifecycle_repository or LifecycleRepository(),
            signal_repository=self._signal_repository,
            risk_engine=self._risk,
            strategy_manager=self._strategy_manager,
            signal_data_source="live",
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
        await self._invalidate_orphan_timeframes()
        await self._cleanup_stale_signals()
        await self._expire_overdue_signals()
        await self._scheduler.start()

        for symbol in self._symbols:
            for timeframe in self._timeframes:
                await self._scheduler.add_interval(
                    name=f"pattern_pipeline_{symbol}_{timeframe}",
                    func=self._pipeline.process_symbol,
                    interval_seconds=self._task_interval_seconds(timeframe),
                    symbol=symbol,
                    timeframe=timeframe,
                )

        await self._scheduler.add_interval(
            name="expire_overdue_signals",
            func=self._expire_overdue_signals,
            interval_seconds=900,
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

    def _task_interval_seconds(self, timeframe: str) -> int:
        """Intervalo de validación adaptado al timeframe (vela/N)."""
        if self._checks_per_candle <= 0:
            return self._interval_seconds
        per_candle = max(timeframe_to_seconds(timeframe) // self._checks_per_candle, 1)
        return max(self._interval_seconds, per_candle)

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

    async def _invalidate_orphan_timeframes(self) -> None:
        """Invalida lifecycles activos de timeframes retirados del pipeline.

        En ciertos casos un timeframe suele resolverse a través de ``market.default_timeframes``
        (velas de tamaño distinto) que no coinciden con ``patterns.lifecycle.timeframes``.
        Para evitar falsas señales, estos lifecycles se invalidan en el arranque.
        """
        try:
            count = await self._pipeline.lifecycle.invalidate_orphans(
                set(self._timeframes),
                reason="timeframe no longer in patterns.lifecycle.timeframes",
            )
        except Exception as e:
            logger.error(f"Failed to invalidate orphan lifecycles: {e}")
            return
        if count:
            logger.info(
                f"Invalidated {count} active lifecycle(s) on timeframes not in pipeline: "
                f"{sorted(set(self._timeframes))}"
            )

    async def _cleanup_stale_signals(self) -> None:
        """Elimina señales live cuyo entry_price no corresponde al mercado actual.

        Limpieza one-time para señales creadas antes de la separación de fuentes
        (data_source): su entry_price proviene de datos históricos y se desvía
        más de ``max_price_deviation`` del último cierre del proveedor live.
        """
        max_deviation = self._settings.patterns.lifecycle.max_price_deviation
        try:
            signals = await self._signal_repository.list(data_source="live", limit=500)
        except Exception as e:
            logger.error(f"Failed to load signals for stale cleanup: {e}")
            return
        if not signals:
            return

        deleted = 0
        for signal in signals:
            if not signal.entry_price:
                continue
            provider = self._resolver(signal.symbol, signal.timeframe)
            if provider is None:
                continue
            try:
                raw = await provider.get_history(
                    symbol=signal.symbol,
                    timeframe=signal.timeframe,
                    limit=1,
                )
            except Exception as e:
                logger.debug(f"Failed to fetch live close for {signal.symbol}: {e}")
                continue
            if not raw:
                continue
            live_close = float(raw[-1].close)
            deviation = abs(live_close - signal.entry_price) / live_close
            if deviation > max_deviation:
                await self._signal_repository.delete(signal.id)
                deleted += 1
                logger.info(
                    f"Removed stale signal {signal.id} for {signal.symbol}: "
                    f"entry {signal.entry_price:.4f} vs live close "
                    f"{live_close:.4f} (deviation {deviation:.1%})"
                )

        if deleted:
            logger.info(f"Stale signal cleanup: removed {deleted} signal(s)")

    async def _expire_overdue_signals(self) -> None:
        """Marca como EXPIRED las señales PENDING cuyo TTL (expires_at) ya venció.

        Corre al arranque y periódicamente vía Scheduler para que las señales
        vencidas no queden eternamente como PENDING (accionables).
        """
        try:
            count = await self._signal_repository.expire_overdue()
        except Exception as e:
            logger.error(f"Failed to expire overdue signals: {e}")
            return
        if count:
            logger.info(f"Expired {count} overdue signal(s)")
