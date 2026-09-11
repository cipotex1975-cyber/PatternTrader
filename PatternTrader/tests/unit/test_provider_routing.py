from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from app.patterns.pipeline import PatternPipeline
from app.patterns.service import (
    PatternService,
    resolve_provider_names,
    resolve_symbol_provider,
)


def _mock_provider(name: str) -> MagicMock:
    provider = MagicMock()
    provider.name = name
    provider.get_history = AsyncMock(return_value=[])
    return provider


def test_resolve_provider_names_dedupes_and_falls_back():
    symbols = ["BTCUSDT", "ETHUSDT", "EURUSD", "USDJPY"]
    providers = {"BTCUSDT": "binance", "EURUSD": "yahoo"}
    names = resolve_provider_names(symbols, providers, "binance")
    assert names == ["binance", "yahoo"]


def test_resolve_symbol_provider_uses_mapping_or_default():
    providers = {"BTCUSDT": "binance", "EURUSD": "yahoo"}
    assert resolve_symbol_provider("BTCUSDT", providers, "binance") == "binance"
    assert resolve_symbol_provider("EURUSD", providers, "binance") == "yahoo"
    assert resolve_symbol_provider("USDJPY", providers, "binance") == "binance"


async def test_pipeline_fetches_from_the_provider_resolved_by_symbol():
    binance = _mock_provider("binance")
    yahoo = _mock_provider("yahoo")

    def resolver(symbol: str, timeframe: str) -> MagicMock:
        return binance if symbol == "BTCUSDT" else yahoo

    pipeline = PatternPipeline(provider_resolver=resolver)

    await pipeline._fetch_candles("BTCUSDT", "1h")
    binance.get_history.assert_awaited_once_with(
        symbol="BTCUSDT", timeframe="1h", limit=pipeline._max_candles
    )
    yahoo.get_history.assert_not_awaited()

    await pipeline._fetch_candles("EURUSD", "1h")
    yahoo.get_history.assert_awaited_once_with(
        symbol="EURUSD", timeframe="1h", limit=pipeline._max_candles
    )


async def test_pipeline_falls_back_to_single_provider_without_resolver():
    provider = _mock_provider("binance")
    pipeline = PatternPipeline(provider=provider)

    pipeline.set_provider_resolver(None)
    await pipeline._fetch_candles("BTCUSDT", "1h")
    provider.get_history.assert_awaited_once_with(
        symbol="BTCUSDT", timeframe="1h", limit=pipeline._max_candles
    )


def test_service_resolver_returns_connected_provider_per_symbol():
    service = PatternService()
    binance = _mock_provider("binance")
    yahoo = _mock_provider("yahoo")
    service._providers = {"binance": binance, "yahoo": yahoo}
    service._provider_route = {"BTCUSDT": "binance", "EURUSD": "yahoo"}

    assert service._resolver("BTCUSDT", "1h") is binance
    assert service._resolver("EURUSD", "1h") is yahoo
    assert service._resolver("USDJPY", "1h") is None
