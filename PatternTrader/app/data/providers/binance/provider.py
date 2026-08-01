from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import ccxt.async_support as ccxt

from app.core.config.settings import get_settings
from app.core.logger import get_logger
from app.data.providers.base import OHLCV, IDataProvider, OrderBook, TickerData
from app.data.providers.factory import DataProviderFactory

logger = get_logger("BinanceProvider")


class BinanceProvider(IDataProvider):
    def __init__(self) -> None:
        settings = get_settings()
        config = settings.data_providers.binance

        exchange_config: dict[str, Any] = {
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }

        if config.api_key:
            exchange_config["apiKey"] = config.api_key
            exchange_config["secret"] = config.api_secret

        if config.testnet:
            exchange_config["sandbox"] = True

        self._exchange = ccxt.binance(exchange_config)
        self._connected = False

    @property
    def name(self) -> str:
        return "binance"

    async def connect(self) -> None:
        if not self._connected:
            await self._exchange.load_markets()
            self._connected = True
            logger.info("Connected to Binance")

    async def disconnect(self) -> None:
        if self._connected:
            await self._exchange.close()
            self._connected = False
            logger.info("Disconnected from Binance")

    async def get_history(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[OHLCV]:
        await self.connect()

        since = int(start.timestamp() * 1000) if start else None
        ohlcv = await self._exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

        candles = []
        for candle in ohlcv:
            candles.append(
                OHLCV(
                    timestamp=datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc),
                    open=candle[1],
                    high=candle[2],
                    low=candle[3],
                    close=candle[4],
                    volume=candle[5],
                )
            )

        return candles

    async def get_live_candle(self, symbol: str, timeframe: str) -> OHLCV:
        await self.connect()
        ohlcv = await self._exchange.fetch_ohlcv(symbol, timeframe, limit=1)
        candle = ohlcv[0]
        return OHLCV(
            timestamp=datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc),
            open=candle[1],
            high=candle[2],
            low=candle[3],
            close=candle[4],
            volume=candle[5],
        )

    async def subscribe(self, symbol: str, timeframe: str) -> AsyncIterator[OHLCV]:
        await self.connect()
        while True:
            try:
                candle = await self.get_live_candle(symbol, timeframe)
                yield candle
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in subscription for {symbol}: {e}")
                await asyncio.sleep(5)

    async def get_ticker(self, symbol: str) -> TickerData:
        await self.connect()
        ticker = await self._exchange.fetch_ticker(symbol)
        return TickerData(
            symbol=symbol,
            last_price=ticker["last"],
            bid=ticker["bid"],
            ask=ticker["ask"],
            volume_24h=ticker["quoteVolume"],
            change_24h=ticker["change"],
            change_24h_pct=ticker["percentage"],
            timestamp=datetime.now(timezone.utc),
        )

    async def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        await self.connect()
        order_book = await self._exchange.fetch_order_book(symbol, limit=depth)
        return OrderBook(
            symbol=symbol,
            bids=order_book["bids"][:depth],
            asks=order_book["asks"][:depth],
            timestamp=datetime.now(timezone.utc),
        )

    async def get_symbols(self) -> list[str]:
        await self.connect()
        return list(self._exchange.markets.keys())

    async def get_exchange_info(self, symbol: str) -> dict[str, Any]:
        await self.connect()
        return self._exchange.markets.get(symbol, {})


DataProviderFactory.register("binance", BinanceProvider)
