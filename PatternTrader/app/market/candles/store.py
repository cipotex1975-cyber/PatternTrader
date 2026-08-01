from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Optional

from app.core.logger import get_logger
from app.market.candles.models import Candle

logger = get_logger("CandleStore")


class CandleStore:
    def __init__(self, max_candles: int = 1000) -> None:
        self._candles: dict[str, dict[str, list[Candle]]] = defaultdict(lambda: defaultdict(list))
        self._max_candles = max_candles

    def add(self, candle: Candle) -> None:
        bucket = self._candles[candle.symbol][candle.timeframe]
        bucket.append(candle)

        if len(bucket) > self._max_candles:
            bucket[:] = bucket[-self._max_candles :]

    def get(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[Candle]:
        candles = self._candles.get(symbol, {}).get(timeframe, [])

        if start:
            candles = [c for c in candles if c.data.timestamp >= start]
        if end:
            candles = [c for c in candles if c.data.timestamp <= end]
        if limit:
            candles = candles[-limit:]

        return candles

    def get_latest(self, symbol: str, timeframe: str) -> Optional[Candle]:
        candles = self._candles.get(symbol, {}).get(timeframe, [])
        return candles[-1] if candles else None

    def get_latest_n(self, symbol: str, timeframe: str, n: int) -> list[Candle]:
        candles = self._candles.get(symbol, {}).get(timeframe, [])
        return candles[-n:]

    def get_all_symbols(self) -> list[str]:
        return list(self._candles.keys())

    def get_all_timeframes(self, symbol: str) -> list[str]:
        return list(self._candles.get(symbol, {}).keys())

    def clear(self, symbol: str | None = None, timeframe: str | None = None) -> None:
        if symbol and timeframe:
            self._candles[symbol][timeframe] = []
        elif symbol:
            self._candles.pop(symbol, None)
        else:
            self._candles.clear()

    def __len__(self) -> int:
        total = 0
        for symbol_candles in self._candles.values():
            for timeframe_candles in symbol_candles.values():
                total += len(timeframe_candles)
        return total
