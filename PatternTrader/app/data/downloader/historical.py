from __future__ import annotations

import asyncio
from datetime import datetime

from app.core.logger import get_logger
from app.data.providers.base import OHLCV, IDataProvider

logger = get_logger("HistoricalDownloader")


class HistoricalDownloader:
    def __init__(self, provider: IDataProvider) -> None:
        self._provider = provider

    async def download(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        batch_size: int = 500,
    ) -> list[OHLCV]:
        all_candles: list[OHLCV] = []
        current_start = start

        logger.info(f"Downloading {symbol} {timeframe} from {start} to {end}")

        while current_start < end:
            candles = await self._provider.get_history(
                symbol=symbol,
                timeframe=timeframe,
                start=current_start,
                end=end,
                limit=batch_size,
            )

            if not candles:
                break

            all_candles.extend(candles)
            current_start = candles[-1].timestamp

            logger.debug(f"Downloaded {len(candles)} candles, total: {len(all_candles)}")

            await asyncio.sleep(0.1)

        logger.info(f"Download complete: {len(all_candles)} candles")
        return all_candles

    async def download_latest(
        self,
        symbol: str,
        timeframe: str,
        count: int = 500,
    ) -> list[OHLCV]:
        return await self._provider.get_history(
            symbol=symbol,
            timeframe=timeframe,
            limit=count,
        )
