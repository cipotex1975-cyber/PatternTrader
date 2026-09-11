from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from app.core.exceptions import DataProviderError
from app.data.providers import (
    AlphaVantageProvider,
    BybitProvider,
    DataProviderFactory,
    InteractiveBrokersProvider,
    MetaTraderProvider,
    PolygonProvider,
    YahooProvider,
)

PROVIDER_NAMES = [
    "binance",
    "bybit",
    "yahoo",
    "polygon",
    "alphavantage",
    "metatrader",
    "interactive_brokers",
]


def test_factory_registers_all_documented_providers():
    registered = set(DataProviderFactory.get_all().keys())
    assert registered == set(PROVIDER_NAMES)


@pytest.mark.parametrize("name", PROVIDER_NAMES)
def test_factory_creates_provider(name):
    provider = DataProviderFactory.create(name)
    assert provider.name == name


class TestBybitProvider:
    def test_normalize_symbol(self):
        assert BybitProvider._normalize_symbol("BTCUSDT") == "BTC/USDT"
        assert BybitProvider._normalize_symbol("ETHBTC") == "ETH/BTC"
        assert BybitProvider._normalize_symbol("BTC/USDT") == "BTC/USDT"

    async def test_get_history(self):
        provider = BybitProvider()
        provider._connected = True
        exchange = AsyncMock()
        exchange.fetch_ohlcv.return_value = [[1700000000000, 100.0, 101.0, 99.0, 100.5, 1000.0]]
        provider._exchange = exchange

        candles = await provider.get_history("BTCUSDT", "1h", limit=10)

        assert len(candles) == 1
        assert candles[0].close == 100.5
        assert candles[0].volume == 1000.0
        exchange.fetch_ohlcv.assert_awaited_once_with("BTC/USDT", "1h", since=None, limit=10)


class TestYahooProvider:
    def test_normalize_symbol(self):
        assert YahooProvider._normalize_symbol("BTCUSDT") == "BTC-USD"
        assert YahooProvider._normalize_symbol("ETHUSDT") == "ETH-USD"
        assert YahooProvider._normalize_symbol("EURUSD") == "EURUSD=X"
        assert YahooProvider._normalize_symbol("GBPUSD") == "GBPUSD=X"
        assert YahooProvider._normalize_symbol("USDJPY") == "USDJPY=X"
        assert YahooProvider._normalize_symbol("AAPL") == "AAPL"

    def test_map_interval(self):
        assert YahooProvider._map_interval("1h") == "1h"
        assert YahooProvider._map_interval("1d") == "1d"

    def test_unsupported_timeframe_raises(self):
        with pytest.raises(DataProviderError):
            YahooProvider._map_interval("2h")

    async def test_get_history(self):
        df = pd.DataFrame(
            {"Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.5], "Volume": [1000.0]},
            index=pd.to_datetime(["2024-01-02"]),
        )
        ticker_mock = MagicMock()
        ticker_mock.history.return_value = df

        with patch("app.data.providers.yahoo.provider.yf.Ticker", return_value=ticker_mock):
            provider = YahooProvider()
            candles = await provider.get_history("AAPL", "1d", limit=10)

        assert len(candles) == 1
        assert candles[0].close == 100.5
        assert candles[0].volume == 1000.0
        assert candles[0].timestamp.tzinfo is not None

    async def test_get_history_4h_aggregates_from_1h(self):
        timestamps = pd.date_range("2024-01-02 00:00:00", periods=6, freq="h", tz="UTC")
        df = pd.DataFrame(
            {
                "Open": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "High": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
                "Low": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                "Close": [1.2, 2.2, 3.2, 4.2, 5.2, 6.2],
                "Volume": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            },
            index=timestamps,
        )
        ticker_mock = MagicMock()
        ticker_mock.history.return_value = df

        with patch("app.data.providers.yahoo.provider.yf.Ticker", return_value=ticker_mock):
            provider = YahooProvider()
            candles = await provider.get_history("EURUSD", "4h", limit=10)

        assert len(candles) == 2
        assert candles[0].open == 1.0
        assert candles[0].high == 4.5
        assert candles[0].low == 0.5
        assert candles[0].close == 4.2
        assert candles[0].volume == 100.0
        assert candles[1].open == 5.0
        assert candles[1].high == 6.5
        assert candles[1].low == 4.5
        assert candles[1].close == 6.2
        assert candles[1].volume == 110.0
        ticker_mock.history.assert_called_once_with(interval="1h", auto_adjust=False, period="10d")

    async def test_order_book_not_supported(self):
        provider = YahooProvider()
        with pytest.raises(DataProviderError):
            await provider.get_order_book("AAPL", depth=10)


class TestPolygonProvider:
    def test_normalize_symbol(self):
        assert PolygonProvider._normalize_symbol("BTCUSDT") == "X:BTCUSDT"
        assert PolygonProvider._normalize_symbol("AAPL") == "AAPL"
        assert PolygonProvider._normalize_symbol("X:BTCUSD") == "X:BTCUSD"

    def test_map_timeframe(self):
        assert PolygonProvider._map_timeframe("4h") == (4, "hour")
        assert PolygonProvider._map_timeframe("1d") == (1, "day")

    def test_unsupported_timeframe_raises(self):
        with pytest.raises(DataProviderError):
            PolygonProvider._map_timeframe("99m")

    async def test_get_history(self):
        provider = PolygonProvider()
        provider._api_key = "test-key"
        provider._connected = True
        provider._client = AsyncMock()

        response = MagicMock()
        response.json.return_value = {
            "status": "OK",
            "results": [
                {
                    "t": 1700000000000,
                    "o": 100.0,
                    "h": 101.0,
                    "l": 99.0,
                    "c": 100.5,
                    "v": 1000.0,
                }
            ],
        }
        provider._client.get.return_value = response

        candles = await provider.get_history("BTCUSDT", "1h", limit=10)

        assert len(candles) == 1
        assert candles[0].close == 100.5
        assert candles[0].volume == 1000.0
        assert candles[0].timestamp == datetime.fromtimestamp(1700000000, tz=timezone.utc)

    async def test_order_book_not_supported(self):
        provider = PolygonProvider()
        with pytest.raises(DataProviderError):
            await provider.get_order_book("AAPL", depth=10)


class TestAlphaVantageProvider:
    def test_is_crypto(self):
        assert AlphaVantageProvider._is_crypto("BTCUSDT") == (True, "BTC", "USDT")
        assert AlphaVantageProvider._is_crypto("AAPL") == (False, "AAPL", "")

    async def test_get_history_stock(self):
        provider = AlphaVantageProvider()
        provider._api_key = "test-key"
        provider._connected = True
        provider._client = AsyncMock()

        response = MagicMock()
        response.json.return_value = {
            "Time Series (Daily)": {
                "2024-01-02": {
                    "1. open": "100.0",
                    "2. high": "101.0",
                    "3. low": "99.0",
                    "4. close": "100.5",
                    "5. volume": "1000",
                }
            }
        }
        provider._client.get.return_value = response

        candles = await provider.get_history("AAPL", "1d", limit=10)

        assert len(candles) == 1
        assert candles[0].close == 100.5
        assert candles[0].volume == 1000.0

    async def test_get_history_crypto(self):
        provider = AlphaVantageProvider()
        provider._api_key = "test-key"
        provider._connected = True
        provider._client = AsyncMock()

        response = MagicMock()
        response.json.return_value = {
            "Time Series (Digital Currency Daily)": {
                "2024-01-02": {
                    "1. open (USD)": "100.0",
                    "2. high (USD)": "101.0",
                    "3. low (USD)": "99.0",
                    "4. close (USD)": "100.5",
                    "5. volume": "1000",
                }
            }
        }
        provider._client.get.return_value = response

        candles = await provider.get_history("BTCUSDT", "1d", limit=10)

        assert len(candles) == 1
        assert candles[0].close == 100.5

    async def test_crypto_intraday_unsupported(self):
        provider = AlphaVantageProvider()
        provider._api_key = "test-key"
        provider._connected = True
        provider._client = AsyncMock()

        with pytest.raises(DataProviderError):
            await provider.get_history("BTCUSDT", "1h", limit=10)

    async def test_order_book_not_supported(self):
        provider = AlphaVantageProvider()
        with pytest.raises(DataProviderError):
            await provider.get_order_book("AAPL", depth=10)


class TestMetaTraderProvider:
    async def test_connect_without_package_raises(self):
        provider = MetaTraderProvider()
        provider._enabled = True
        with pytest.raises(DataProviderError):
            await provider.connect()

    def test_unsupported_timeframe_raises(self):
        with pytest.raises(DataProviderError):
            MetaTraderProvider._map_timeframe("99m")


class TestInteractiveBrokersProvider:
    async def test_connect_without_package_raises(self):
        provider = InteractiveBrokersProvider()
        provider._enabled = True
        with pytest.raises(DataProviderError):
            await provider.connect()

    def test_unsupported_timeframe_raises(self):
        with pytest.raises(DataProviderError):
            InteractiveBrokersProvider._map_bar_size("99m")
