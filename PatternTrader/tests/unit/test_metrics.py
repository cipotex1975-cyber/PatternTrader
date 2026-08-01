from datetime import datetime, timedelta, timezone

from app.backtesting.metrics import MetricsCalculator
from app.backtesting.models import (
    Trade,
    TradeDirection,
    TradeStatus,
)


def _trade(i: int, pnl: float) -> Trade:
    entry = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
    return Trade(
        id=str(i),
        symbol="BTCUSDT",
        timeframe="1h",
        direction=TradeDirection.LONG,
        entry_price=100.0,
        entry_time=entry,
        exit_time=entry + timedelta(days=1),
        stop_loss=95.0,
        take_profit=110.0,
        size=1.0,
        pnl=pnl,
        pnl_pct=pnl / 100.0,
        status=TradeStatus.CLOSED,
    )


def _equity_curve(values):
    return [
        {"timestamp": (datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)).isoformat(),
         "equity": v, "capital": v, "open_pnl": 0.0}
        for i, v in enumerate(values)
    ]


def test_metrics_calculates_core():
    trades = [
        _trade(0, 10.0),
        _trade(1, 10.0),
        _trade(2, -5.0),
        _trade(3, 10.0),
        _trade(4, -5.0),
    ]
    curve = _equity_curve([100, 110, 120, 115, 125, 135])
    metrics = MetricsCalculator.calculate(
        trades,
        curve,
        initial_capital=100,
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 1, 10, tzinfo=timezone.utc),
    )

    assert metrics.total_trades == 5
    assert metrics.winning_trades == 3
    assert metrics.losing_trades == 2
    assert metrics.win_rate == 0.6
    assert metrics.profit_factor == 30.0 / 10.0
    assert metrics.expectancy > 0
    assert metrics.max_drawdown_pct > 0
    assert metrics.ulcer_index >= 0


def test_metrics_calmar_ulcer_sortino_populated():
    trades = [_trade(i, 3.0 if i % 2 == 0 else -1.0) for i in range(20)]
    curve = _equity_curve([100, 102, 101, 104, 103, 106, 98, 105, 110, 108, 112, 109, 115, 113, 117, 114, 119, 116, 121, 118, 123])
    metrics = MetricsCalculator.calculate(
        trades,
        curve,
        initial_capital=100,
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 2, 1, tzinfo=timezone.utc),
    )
    assert metrics.sortino_ratio != 0.0
    assert metrics.calmar_ratio != 0.0
    assert metrics.ulcer_index > 0
    assert metrics.max_drawdown_pct > 0
    assert metrics.annual_return > 0


def test_metrics_empty_returns_defaults():
    metrics = MetricsCalculator.calculate([], [], initial_capital=100, start_date=datetime.utcnow(), end_date=datetime.utcnow())
    assert metrics.total_trades == 0
    assert metrics.win_rate == 0.0


def test_classification_metrics():
    cm = MetricsCalculator.classification_metrics(
        y_true=[1, 1, 0, 0, 1, 0],
        y_pred=[1, 0, 0, 0, 1, 1],
        y_proba=[0.9, 0.4, 0.3, 0.2, 0.8, 0.7],
    )
    assert cm.true_positives == 2
    assert cm.false_positives == 1
    assert cm.false_negatives == 1
    assert cm.confusion_matrix == [[2, 1], [1, 2]]
    assert 0 < cm.precision <= 1
    assert 0 < cm.recall <= 1
    assert 0 < cm.f1_score <= 1
    assert 0.5 < cm.roc_auc <= 1.0


def test_classification_metrics_empty():
    cm = MetricsCalculator.classification_metrics([], [])
    assert cm.accuracy == 0.0
    assert cm.confusion_matrix == [[0, 0], [0, 0]]
