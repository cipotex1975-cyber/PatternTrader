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
