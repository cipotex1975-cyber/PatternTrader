from datetime import datetime, timedelta, timezone

from app.backtesting.models import BacktestConfig, TradeStatus
from app.backtesting.runner import BacktestRunner
from app.market.candles.models import Candle, CandleData
from app.patterns.base_pattern import PatternResult, PatternStatus, PatternType, TradeDirection


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


def _candles(n: int = 200) -> list[Candle]:
    return [_candle(100.0 + i * 0.1, i) for i in range(n)]


def _pattern(entry: float, stop: float, target: float) -> PatternResult:
    return PatternResult(
        pattern_name="double_bottom",
        pattern_type=PatternType.REVERSAL,
        symbol="BTCUSDT",
        timeframe="1h",
        direction=TradeDirection.LONG,
        status=PatternStatus.CONFIRMED,
        confidence=0.8,
        entry_price=entry,
        stop_loss=stop,
        take_profit=target,
        detected_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def _pattern_spec(entry: float, stop: float, target: float) -> dict:
    return {
        "name": "spec",
        "candles": _candles(),
        "patterns": [_pattern(entry, stop, target)],
    }


def test_runner_simple_backtest():
    runner = BacktestRunner(BacktestConfig(initial_capital=10000))
    result = runner.run(_candles(), [_pattern(100.0, 95.0, 110.0)])
    assert result.metadata["name"] == "simple"
    assert result.metrics.total_trades >= 0
    assert result.initial_capital == 10000


def test_runner_run_multiple():
    runner = BacktestRunner()
    specs = [
        {"name": "a", "candles": _candles(), "patterns": [_pattern(100.0, 95.0, 110.0)]},
        {"name": "b", "candles": _candles(), "patterns": []},
    ]
    results = runner.run_multiple(specs)
    assert len(results) == 2
    assert results[0].metadata["name"] == "a"
    assert results[1].metadata["name"] == "b"


def test_runner_run_parallel():
    runner = BacktestRunner()
    specs = [_pattern_spec(100.0, 95.0, 110.0), _pattern_spec(101.0, 96.0, 111.0)]
    results = runner.run_parallel(specs, max_workers=2)
    assert len(results) == 2
    assert all(r.metrics.total_trades >= 0 for r in results)


def test_runner_compare_ranks_by_return():
    runner = BacktestRunner()
    results = runner.run_multiple(
        [
            {"name": "winner", "candles": _candles(), "patterns": [_pattern(100.0, 95.0, 130.0)]},
            {"name": "loser", "candles": _candles(), "patterns": [_pattern(100.0, 99.9, 100.1)]},
        ]
    )
    summary = runner.compare(results)
    assert summary["by_return"][0]["name"] == "winner"
    assert len(summary["by_return"]) == 2


def test_runner_trade_is_closed():
    runner = BacktestRunner(BacktestConfig(initial_capital=10000))
    result = runner.run(_candles(300), [_pattern(100.0, 90.0, 130.0)])
    assert result.trades
    assert all(t.status == TradeStatus.CLOSED for t in result.trades)
