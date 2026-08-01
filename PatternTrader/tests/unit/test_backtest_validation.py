from datetime import datetime, timedelta

from app.backtesting.models import Trade, TradeDirection, TradeStatus
from app.backtesting.validation import (
    CrossValidator,
    MonteCarloSimulator,
    OutOfSampleValidator,
    RollingWindowValidator,
    TimeSeriesSplitter,
    WalkForwardValidator,
)


class FakeCandle:
    def __init__(self, i: int) -> None:
        self.data = type("D", (), {"timestamp": datetime(2024, 1, 1) + timedelta(days=i)})()


def _candles(n: int) -> list:
    return [FakeCandle(i) for i in range(n)]


def _trade(i: int, pnl: float) -> Trade:
    base = datetime(2024, 1, 1)
    return Trade(
        id=str(i),
        symbol="BTCUSDT",
        timeframe="1h",
        direction=TradeDirection.LONG,
        entry_price=100.0,
        entry_time=base,
        exit_time=base + timedelta(days=1),
        pnl=pnl,
        pnl_pct=pnl / 100,
        status=TradeStatus.CLOSED,
    )


def test_walk_forward_splits():
    folds = TimeSeriesSplitter.walk_forward(1000, train_size=300, test_size=100, step=100)
    assert len(folds) == 7
    assert folds[0]["train_start"] == 0
    assert folds[0]["train_end"] == 300
    assert folds[0]["test_start"] == 300
    assert folds[0]["test_end"] == 400
    assert folds[-1]["test_end"] <= 1000


def test_rolling_window_splits():
    folds = TimeSeriesSplitter.rolling_window(500, window_size=200, step=100)
    assert len(folds) == 4
    assert all(f["test_size"] == 200 for f in folds)


def test_train_test_split_oos():
    split = TimeSeriesSplitter.train_test_split(100, test_ratio=0.3)
    assert split["train_size"] == 70
    assert split["test_size"] == 30


def test_kfold_splits_cover_all():
    folds = TimeSeriesSplitter.kfold(10, n_splits=5)
    assert len(folds) == 5
    assert all(f["test_size"] > 0 for f in folds)


def _evaluate_with_sharpe(candles):
    return {
        "total_return": len(candles) / 100.0,
        "sharpe_ratio": len(candles) / 100.0,
        "win_rate": 0.5,
    }


def test_walk_forward_validator_runs():
    validator = WalkForwardValidator(
        train_size=300, test_size=100, step=100, evaluate_fn=_evaluate_with_sharpe
    )
    result = validator.run(_candles(1000))
    assert result.method == "walk_forward"
    assert len(result.folds) == 7
    assert "sharpe_ratio" in result.aggregate
    assert result.aggregate["sharpe_ratio"]["mean"] > 0


def test_oos_validator_runs():
    validator = OutOfSampleValidator(test_ratio=0.3, evaluate_fn=_evaluate_with_sharpe)
    result = validator.run(_candles(100))
    assert len(result.folds) == 1
    assert result.folds[0].metrics["total_return"] > 0


def test_cross_validator_runs():
    validator = CrossValidator(n_splits=4, evaluate_fn=_evaluate_with_sharpe)
    result = validator.run(_candles(80))
    assert len(result.folds) == 4


def test_rolling_validator_runs():
    validator = RollingWindowValidator(window_size=200, step=50, evaluate_fn=_evaluate_with_sharpe)
    result = validator.run(_candles(500))
    assert len(result.folds) == 7


def test_monte_carlo_simulates():
    trades = [_trade(i, 100.0 if i % 2 == 0 else -50.0) for i in range(20)]
    sim = MonteCarloSimulator(random_state=42).simulate(
        trades, n_simulations=200, initial_capital=100000
    )
    assert sim.simulations == 200
    assert len(sim.final_equities) == 200
    assert sim.median_final_equity > 0
    assert "p5" in sim.percentiles and "p95" in sim.percentiles
    assert 0 <= sim.probability_of_profit <= 1
    assert sim.var_95 <= 0


def test_monte_carlo_empty_trades():
    sim = MonteCarloSimulator().simulate([], n_simulations=50)
    assert sim.simulations == 0
