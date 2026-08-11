from datetime import datetime, timedelta, timezone

from app.backtesting.models import TradeStatus
from app.database.repositories import LifecycleRepository, SignalRepository, TradeRepository
from app.execution.engine import ExecutionEngine
from app.market.candles.models import Candle, CandleData
from app.patterns.pipeline import PatternPipeline
from app.risk.engine import RiskEngine

from ..conftest import requires_postgres


def _double_top_candles() -> list[Candle]:
    closes = [47000 + i * 300 for i in range(10)]
    closes += [49400, 49100, 48800]
    closes += [49100, 49400, 49700, 49950, 49700, 49400]
    closes += [48900, 48700, 48500, 48300, 48100]
    volumes = [1000] * len(closes)
    for i in range(-5, 0):
        volumes[i] = 2000

    candles = []
    prev = closes[0]
    base_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i, (close, volume) in enumerate(zip(closes, volumes)):
        candles.append(
            Candle(
                symbol="BTCUSDT",
                timeframe="1h",
                data=CandleData(
                    timestamp=base_ts + timedelta(hours=i),
                    open=prev,
                    high=max(prev, close) * 1.002,
                    low=min(prev, close) * 0.998,
                    close=close,
                    volume=volume,
                ),
            )
        )
        prev = close
    return candles


@requires_postgres
async def test_full_flow_persist_and_restart(pg_db):
    lifecycle_repo = LifecycleRepository()
    signal_repo = SignalRepository()
    trade_repo = TradeRepository()

    pipeline = PatternPipeline(
        data_source=lambda symbol, timeframe: _double_top_candles(),
        lifecycle_repository=lifecycle_repo,
        signal_repository=signal_repo,
        strategy_params={"breakout": {"rsi_min": 25.0, "rsi_max": 80.0}},
    )
    for _ in range(4):
        await pipeline.process_symbol("BTCUSDT", "1h")

    signals = await signal_repo.list(symbol="BTCUSDT", limit=50)
    assert len(signals) >= 1
    signal = signals[0]
    assert signal.pattern_name == "double_top"

    lcs = pipeline.lifecycle.get_all()
    signal_lc = next(lc for lc in lcs if lc.current_state.value == "SIGNAL_SENT")

    execution = ExecutionEngine(
        lifecycle=pipeline.lifecycle,
        repository=trade_repo,
        risk_engine=RiskEngine(),
    )
    trade = await execution.open_trade(
        {
            "symbol": signal.symbol,
            "timeframe": signal.timeframe,
            "direction": signal.direction,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "size": 0.5,
            "pattern_name": signal.pattern_name,
            "score": signal.score,
            "signal_id": str(signal.id),
            "pattern_id": str(signal_lc.pattern_id),
        }
    )
    assert trade is not None
    assert trade.status.value == "OPEN"

    stored_trade = await trade_repo.get(trade.id)
    assert stored_trade is not None
    assert stored_trade.symbol == "BTCUSDT"
    assert stored_trade.entry_time.tzinfo is not None

    rows = await lifecycle_repo.list(limit=50)
    assert len(rows) >= 1

    restarted = PatternPipeline(
        lifecycle_repository=lifecycle_repo,
        signal_repository=signal_repo,
    )
    loaded = await restarted.lifecycle.rehydrate(rows)
    assert loaded >= 1

    rehydrated_signals = await signal_repo.list(symbol="BTCUSDT", limit=50)
    assert any(s.id == signal.id for s in rehydrated_signals)

    rehydrated_trades = await trade_repo.list(status=TradeStatus.OPEN)
    assert any(t.id == trade.id for t in rehydrated_trades)

    state = restarted.lifecycle.get_all()
    assert any(lc.current_state.value == "OPEN" for lc in state)
