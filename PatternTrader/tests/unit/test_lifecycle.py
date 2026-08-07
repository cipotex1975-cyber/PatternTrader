import pytest

from app.lifecycle.engine import LifecycleEngine
from app.lifecycle.models import LifecycleState
from app.patterns.base_pattern import PatternResult, PatternType


def create_test_pattern():
    return PatternResult(
        pattern_name="double_top",
        pattern_type=PatternType.REVERSAL,
        symbol="BTCUSDT",
        timeframe="1h",
        confidence=0.85,
    )


@pytest.mark.asyncio
async def test_lifecycle_registration():
    engine = LifecycleEngine()
    pattern = create_test_pattern()

    lifecycle = await engine.register(pattern)
    assert lifecycle is not None
    assert lifecycle.current_state == LifecycleState.DETECTED
    assert lifecycle.symbol == "BTCUSDT"


@pytest.mark.asyncio
async def test_lifecycle_transition():
    engine = LifecycleEngine()
    pattern = create_test_pattern()

    lifecycle = await engine.register(pattern)
    transition = await engine.transition(
        lifecycle.id,
        LifecycleState.FORMING,
        reason="Pattern structure detected",
    )

    assert transition is not None
    assert transition.from_state == LifecycleState.DETECTED
    assert transition.to_state == LifecycleState.FORMING


@pytest.mark.asyncio
async def test_lifecycle_get_active():
    engine = LifecycleEngine()
    pattern = create_test_pattern()

    await engine.register(pattern)
    active = engine.get_active()
    assert len(active) == 1


@pytest.mark.asyncio
async def test_lifecycle_statistics():
    engine = LifecycleEngine()
    pattern = create_test_pattern()

    await engine.register(pattern)
    stats = engine.get_statistics()
    assert stats["DETECTED"] == 1


@pytest.mark.asyncio
async def test_transition_by_pattern_id():
    engine = LifecycleEngine()
    pattern = create_test_pattern()
    lifecycle = await engine.register(pattern)

    result = await engine.transition_by_pattern(
        pattern.id, LifecycleState.FORMING, "structure detected"
    )

    assert result is not None
    assert result.to_state == LifecycleState.FORMING
    assert lifecycle.current_state == LifecycleState.FORMING


@pytest.mark.asyncio
async def test_transition_by_pattern_id_unknown_returns_none():
    engine = LifecycleEngine()
    await engine.register(create_test_pattern())

    result = await engine.transition_by_pattern(
        "00000000-0000-0000-0000-000000000000",
        LifecycleState.FORMING,
        "should not exist",
    )
    assert result is None


@pytest.mark.asyncio
async def test_lifecycle_reaches_closed_via_open_and_tp():
    engine = LifecycleEngine()
    pattern = create_test_pattern()
    lifecycle = await engine.register(pattern)

    await engine.transition_by_pattern(pattern.id, LifecycleState.FORMING, "forming")
    await engine.transition_by_pattern(pattern.id, LifecycleState.WAITING_BREAKOUT, "waiting")
    await engine.transition_by_pattern(pattern.id, LifecycleState.CONFIRMED, "confirmed")
    await engine.transition_by_pattern(pattern.id, LifecycleState.SIGNAL_SENT, "signal")
    await engine.transition_by_pattern(pattern.id, LifecycleState.OPEN, "trade opened")
    await engine.transition_by_pattern(pattern.id, LifecycleState.TP_HIT, "tp hit")
    await engine.transition_by_pattern(pattern.id, LifecycleState.CLOSED, "closed")

    assert lifecycle.current_state == LifecycleState.CLOSED
    states = [t.to_state for t in lifecycle.transitions]
    assert LifecycleState.OPEN in states
    assert LifecycleState.TP_HIT in states
    assert LifecycleState.CLOSED in states


@pytest.mark.asyncio
async def test_lifecycle_cancelled_and_rejected_not_active():
    engine = LifecycleEngine()
    for status in (LifecycleState.CANCELLED, LifecycleState.REJECTED, LifecycleState.EXPIRED):
        pattern = create_test_pattern()
        await engine.register(pattern)
        await engine.transition_by_pattern(pattern.id, status, f"to {status.value}")

    assert engine.get_active() == []
    for status in (LifecycleState.CANCELLED, LifecycleState.REJECTED, LifecycleState.EXPIRED):
        assert len(engine.get_by_state(status)) == 1


@pytest.mark.asyncio
async def test_lifecycle_rehydrate_from_db(sync_db):
    from app.database.repositories import LifecycleRepository

    repo = LifecycleRepository()
    pattern = create_test_pattern()
    lifecycle = await LifecycleEngine(repository=repo).register(pattern)

    engine = LifecycleEngine(repository=repo)
    loaded = await engine.rehydrate_from_db()
    assert loaded == 1

    restored = engine.get_by_pattern(pattern.id)
    assert restored is not None
    assert restored.id == lifecycle.id
    assert restored.symbol == "BTCUSDT"
    assert restored.current_state == LifecycleState.DETECTED
