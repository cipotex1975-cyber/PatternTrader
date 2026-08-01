from __future__ import annotations

from app.core.config.settings import get_settings
from app.core.logger import get_logger
from app.data.providers.base import IDataProvider
from app.data.providers.factory import DataProviderFactory
from app.patterns.pipeline import PatternPipeline
from app.scheduler.main import Scheduler

logger = get_logger("PatternService")


class PatternService:
    """Ejecuta el pipeline de patrones de forma periódica para cada símbolo/timeframe."""

    def __init__(self) -> None:
        settings = get_settings()
        lifecycle_settings = settings.patterns.lifecycle
        self._enabled = lifecycle_settings.enabled
        self._interval_seconds = lifecycle_settings.check_interval_seconds
        self._symbols = settings.market.default_symbols
        self._timeframes = lifecycle_settings.timeframes
        self._candle_limit = lifecycle_settings.candle_limit

        self._provider: IDataProvider | None = None
        self._pipeline = PatternPipeline(max_candles=self._candle_limit)
        self._scheduler = Scheduler()

    async def start(self) -> None:
        if not self._enabled:
            logger.info("PatternService disabled; pipeline not started")
            return

        try:
            self._provider = DataProviderFactory.create()
            await self._provider.connect()
            self._pipeline.attach_provider(self._provider)
        except Exception as e:
            logger.error(f"Failed to connect data provider: {e}")
            self._provider = None

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
        if self._provider is not None:
            try:
                await self._provider.disconnect()
            except Exception as e:
                logger.error(f"Failed to disconnect provider: {e}")
        logger.info("PatternService stopped")

    def get_scheduler_tasks(self) -> list[str]:
        return self._scheduler.get_tasks()
