from datetime import datetime, timedelta, timezone

from app.backtesting.models import BacktestConfig, TradeStatus
from app.backtesting.runner import BacktestRunner
from app.database.repositories import BacktestRepository
from app.market.candles.models import Candle, CandleData
from app.patterns.base_pattern import PatternResult, PatternStatus, PatternType, TradeDirection

from ..conftest import requires_postgres


def _candle(close: float, index: int) -> Candle:
    return Candle(
        symbol="BTCUSDT",
        timeframe="1h",
        data=CandleData(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=index),
            open=close,
            high=close * 1.02,
            low=close * 0.98,
            close=close,
            volume=10000.0,
        ),
    )


def _candles(n: int = 300) -> list[Candle]:
    return [_candle(100.0 + i * 0.1, i) for i in range(n)]


def _pattern() -> PatternResult:
    return PatternResult(
        pattern_name="double_bottom",
        pattern_type=PatternType.REVERSAL,
        symbol="BTCUSDT",
        timeframe="1h",
        direction=TradeDirection.LONG,
        status=PatternStatus.CONFIRMED,
        confidence=0.8,
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=130.0,
        detected_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


@requires_postgres
async def test_backtest_roundtrip_persisted(pg_db):
    result = BacktestRunner(
        BacktestConfig(initial_capital=10000, commission=0.001)
    ).run(_candles(), [_pattern()])

    repo = BacktestRepository()
    backtest_id = await repo.add(result, name="integration-roundtrip")

    item = await repo.get(backtest_id)
    assert item is not None
    assert item["name"] == "integration-roundtrip"

    stored = item["result"]
    assert stored.initial_capital == 10000
    assert stored.metrics.total_trades == result.metrics.total_trades
    assert stored.metrics.total_trades > 0
    assert stored.trades
    assert all(t.status == TradeStatus.CLOSED for t in stored.trades)
    assert len(stored.trades) == len(result.trades)
    assert stored.equity_curve == result.equity_curve
    assert stored.config.initial_capital == 10000


@requires_postgres
async def test_backtest_list_orders_by_recency(pg_db):
    repo = BacktestRepository()
    first = await repo.add(
        BacktestRunner(BacktestConfig(initial_capital=10000)).run(_candles(), [_pattern()]),
        name="first",
    )
    second = await repo.add(
        BacktestRunner(BacktestConfig(initial_capital=20000)).run(_candles(), [_pattern()]),
        name="second",
    )

    items = await repo.list(limit=10)
    names = [item["name"] for item in items]
    assert "second" in names and "first" in names
    assert names.index("second") < names.index("first")
    assert any(item["id"] == first for item in items)
    assert any(item["id"] == second for item in items)
