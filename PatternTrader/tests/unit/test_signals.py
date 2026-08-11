from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest

from app.core.config.settings import get_settings
from app.patterns.base_pattern import PatternResult, PatternType
from app.scoring.models import ScoreResult
from app.signals.engine import SignalEngine
from app.signals.models import Signal, SignalPriority, SignalStatus


def make_pattern(**overrides: Any) -> PatternResult:
    data: dict[str, Any] = dict(
        pattern_name="double_top",
        pattern_type=PatternType.REVERSAL,
        symbol="BTCUSDT",
        timeframe="1h",
        confidence=0.85,
        health=90.0,
        entry_price=50000.0,
        stop_loss=51000.0,
        take_profit=48000.0,
    )
    data.update(overrides)
    return PatternResult(**data)


def make_score(total: float) -> ScoreResult:
    return ScoreResult(total_score=total, grade="A", confidence=0.9)


class FakeSignalRepository:
    def __init__(self) -> None:
        self.signals: dict[str, Signal] = {}
        self.updates: list[str] = []

    async def add(self, signal: Signal) -> None:
        self.signals[str(signal.id)] = signal

    async def update_status(self, signal_id, status, sent_at=None):
        self.updates.append(str(status.value))


@pytest.mark.asyncio
async def test_create_signal_critical_priority():
    engine = SignalEngine()
    signal = await engine.create_signal(make_pattern(), make_score(96.0), ml_probability=0.8)

    assert signal is not None
    assert signal.priority == SignalPriority.CRITICAL
    assert signal.direction == "SHORT"
    assert signal.symbol == "BTCUSDT"
    assert signal.entry_price == 50000.0
    assert signal.risk_reward_ratio == pytest.approx(2.0)
    assert signal.ml_probability == 0.8
    assert "ML probability" in " ".join(signal.reasons)
    assert signal.expires_at is not None


@pytest.mark.asyncio
async def test_create_signal_low_score_returns_none():
    engine = SignalEngine()
    signal = await engine.create_signal(make_pattern(), make_score(50.0))
    assert signal is None


@pytest.mark.asyncio
async def test_create_signal_priority_bands():
    engine = SignalEngine()
    bands = [(95.0, SignalPriority.CRITICAL), (85.0, SignalPriority.HIGH),
             (75.0, SignalPriority.MEDIUM), (60.0, SignalPriority.LOW)]
    for i, (score, expected) in enumerate(bands):
        signal = await engine.create_signal(make_pattern(symbol=f"PAIR{i}"), make_score(score))
        assert signal is not None
        assert signal.priority == expected


@pytest.mark.asyncio
async def test_create_signal_cooldown_dedup():
    engine = SignalEngine()
    first = await engine.create_signal(make_pattern(), make_score(96.0))
    assert first is not None
    second = await engine.create_signal(make_pattern(), make_score(96.0))
    assert second is None


@pytest.mark.asyncio
async def test_create_signal_invalid_prices_returns_none():
    engine = SignalEngine()
    signal = await engine.create_signal(
        make_pattern(entry_price=None), make_score(96.0)
    )
    assert signal is None


@pytest.mark.asyncio
async def test_create_signal_persists_to_repository():
    repo = FakeSignalRepository()
    engine = SignalEngine(repository=repo)
    signal = await engine.create_signal(make_pattern(), make_score(96.0))
    assert signal is not None
    assert str(signal.id) in repo.signals


@pytest.mark.asyncio
async def test_mark_sent_delivered_failed():
    repo = FakeSignalRepository()
    engine = SignalEngine(repository=repo)
    signal = await engine.create_signal(make_pattern(), make_score(96.0))
    assert signal is not None

    sent = await engine.mark_sent(signal.id)
    assert sent.status == SignalStatus.SENT
    assert sent.sent_at is not None

    delivered = await engine.mark_delivered(signal.id)
    assert delivered.status == SignalStatus.DELIVERED

    failed = await engine.mark_failed(signal.id, reason="broker down")
    assert failed.status == SignalStatus.FAILED
    assert failed.metadata["failure_reason"] == "broker down"
    assert len(repo.updates) == 3


@pytest.mark.asyncio
async def test_mark_unknown_signal_returns_none():
    engine = SignalEngine()
    assert await engine.mark_sent(uuid4()) is None


@pytest.mark.asyncio
async def test_clear_cooldown_allows_resend():
    engine = SignalEngine()
    await engine.create_signal(make_pattern(), make_score(96.0))
    assert await engine.create_signal(make_pattern(), make_score(96.0)) is None
    engine.clear_cooldown()
    assert await engine.create_signal(make_pattern(), make_score(96.0)) is not None


def test_cooldown_minutes_reads_from_settings():
    engine = SignalEngine()
    assert engine._cooldown_minutes == get_settings().patterns.scoring.cooldown_minutes


@pytest.mark.asyncio
async def test_dedup_persists_across_engine_instances():
    engine = SignalEngine()
    assert await engine.create_signal(make_pattern(), make_score(96.0)) is not None

    restarted = SignalEngine()
    assert await restarted.create_signal(make_pattern(), make_score(96.0)) is None


def test_signal_model_priority_score():
    low = SignalPriority.LOW
    assert Signal(symbol="S", timeframe="1h", pattern_name="p", direction="LONG",
                  priority=low, entry_price=1.0, stop_loss=2.0, take_profit=3.0,
                  risk_reward_ratio=1.0, score=50.0, health=50.0, ml_probability=0.5
                  ).priority_score == 25


def test_signal_is_expired():
    signal = Signal(
        symbol="S", timeframe="1h", pattern_name="p", direction="LONG",
        priority=SignalPriority.MEDIUM, entry_price=1.0, stop_loss=2.0,
        take_profit=3.0, risk_reward_ratio=1.0, score=50.0, health=50.0,
        ml_probability=0.5,
        expires_at=datetime.utcnow() - timedelta(hours=1),
    )
    assert signal.is_expired


@pytest.mark.asyncio
async def test_mark_sent_skips_expired_signal():
    repo = FakeSignalRepository()
    engine = SignalEngine(repository=repo)
    signal = await engine.create_signal(make_pattern(), make_score(96.0))
    assert signal is not None
    signal.expires_at = datetime.utcnow() - timedelta(hours=1)

    sent = await engine.mark_sent(signal.id)
    assert sent is None
    assert signal.status == SignalStatus.FAILED
    assert signal.metadata["failure_reason"] == "signal expired"


def test_signal_ttl_hours_from_settings():
    engine = SignalEngine()
    assert (
        engine._scoring_config.signal_ttl_hours
        == get_settings().patterns.scoring.signal_ttl_hours
    )


@pytest.mark.asyncio
async def test_create_signal_expires_at_uses_configured_ttl():
    engine = SignalEngine()
    signal = await engine.create_signal(make_pattern(), make_score(96.0))
    assert signal is not None
    expected = signal.created_at + timedelta(hours=get_settings().patterns.scoring.signal_ttl_hours)
    assert abs((signal.expires_at - expected).total_seconds()) < 1
