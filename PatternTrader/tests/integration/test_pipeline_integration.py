from datetime import datetime, timedelta, timezone

from app.database.repositories import LifecycleRepository, SignalRepository
from app.market.candles.models import Candle, CandleData
from app.patterns.pipeline import PatternPipeline

from ..conftest import requires_postgres


def _build_candles() -> list[Candle]:
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
async def test_pipeline_detection_persists_lifecycle(pg_db):
    pipeline = PatternPipeline(
        data_source=lambda symbol, timeframe: _build_candles(),
        lifecycle_repository=LifecycleRepository(),
        signal_repository=SignalRepository(),
        strategy_params={"breakout": {"rsi_min": 25.0, "rsi_max": 80.0}},
    )

    stats = await pipeline.process_symbol("BTCUSDT", "1h")
    assert stats["tracked"] >= 1

    for _ in range(4):
        await pipeline.process_symbol("BTCUSDT", "1h")

    repo = LifecycleRepository()
    rows = await repo.list(limit=50)
    assert len(rows) >= 1

    pattern_result, lifecycle = rows[0]
    assert pattern_result.symbol == "BTCUSDT"
    assert lifecycle.symbol == "BTCUSDT"
    assert lifecycle.is_active


@requires_postgres
async def test_pipeline_signal_persisted(pg_db):
    pipeline = PatternPipeline(
        data_source=lambda symbol, timeframe: _build_candles(),
        lifecycle_repository=LifecycleRepository(),
        signal_repository=SignalRepository(),
        strategy_params={"breakout": {"rsi_min": 25.0, "rsi_max": 80.0}},
    )

    for _ in range(4):
        await pipeline.process_symbol("BTCUSDT", "1h")

    repo = SignalRepository()
    signals = await repo.list(symbol="BTCUSDT", limit=50)
    assert len(signals) >= 1
    assert signals[0].pattern_name == "double_top"


@requires_postgres
async def test_pipeline_restart_rehydrates_from_db(pg_db):
    pipeline = PatternPipeline(
        data_source=lambda symbol, timeframe: _build_candles(),
        lifecycle_repository=LifecycleRepository(),
        signal_repository=SignalRepository(),
        strategy_params={"breakout": {"rsi_min": 25.0, "rsi_max": 80.0}},
    )

    for _ in range(4):
        await pipeline.process_symbol("BTCUSDT", "1h")

    rows = await LifecycleRepository().list(limit=50)
    assert len(rows) >= 1
    persisted_id = rows[0][1].id
    persisted_state = rows[0][1].current_state

    restarted = PatternPipeline(
        lifecycle_repository=LifecycleRepository(),
        signal_repository=SignalRepository(),
    )
    rehydrated = await restarted.lifecycle.rehydrate(rows)
    assert rehydrated >= 1

    lifecycles = restarted.lifecycle.get_all()
    by_id = {lc.id: lc for lc in lifecycles}
    assert persisted_id in by_id
    assert by_id[persisted_id].current_state == persisted_state
