from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, AsyncIterator

from pydantic import BaseModel


class OHLCV(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    model_config = {"frozen": True}


class TickerData(BaseModel):
    symbol: str
    last_price: float
    bid: float
    ask: float
    volume_24h: float
    change_24h: float
    change_24h_pct: float
    timestamp: datetime


class OrderBook(BaseModel):
    symbol: str
    bids: list[list[float]]
    asks: list[list[float]]
    timestamp: datetime


class IDataProvider(ABC):
    """Abstract interface for data providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        ...

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the provider."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the provider."""
        ...

    @abstractmethod
    async def get_history(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[OHLCV]:
        """Get historical OHLCV data."""
        ...

    @abstractmethod
    async def get_live_candle(self, symbol: str, timeframe: str) -> OHLCV:
        """Get the latest candle."""
        ...

    @abstractmethod
    async def subscribe(self, symbol: str, timeframe: str) -> AsyncIterator[OHLCV]:
        """Subscribe to real-time candle updates."""
        ...

    @abstractmethod
    async def get_ticker(self, symbol: str) -> TickerData:
        """Get current ticker data."""
        ...

    @abstractmethod
    async def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        """Get order book data."""
        ...

    @abstractmethod
    async def get_symbols(self) -> list[str]:
        """Get available trading symbols."""
        ...

    @abstractmethod
    async def get_exchange_info(self, symbol: str) -> dict[str, Any]:
        """Get exchange information for a symbol."""
        ...
