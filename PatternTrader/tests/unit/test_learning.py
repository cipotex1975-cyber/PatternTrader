from datetime import datetime, timedelta

import pytest

from app.backtesting.models import Trade, TradeDirection, TradeStatus
from app.learning.features import FeatureBuilder
from app.learning.models import KnowledgeEntry, LearningMode, TradeOutcome
from app.learning.offline import OfflineLearner
from app.learning.online import OnlineLearner
from app.learning.repository import MemoryKnowledgeRepository
from app.learning.service import LearningService

_INDICATORS = {
    "rsi": 45.0,
    "atr": 100.0,
    "macd": 3.0,
    "macd_histogram": 0.5,
    "momentum": 1.2,
    "bb_upper": 51000.0,
    "bb_lower": 49000.0,
    "volume": 5000.0,
}


def _closed_trade(i: int, win: bool) -> Trade:
    base = datetime(2024, 1, 1) + timedelta(days=i)
    return Trade(
        id=str(i),
        symbol="BTCUSDT",
        timeframe="1h",
        direction=TradeDirection.LONG,
        entry_price=50000.0,
        entry_time=base,
        exit_time=base + timedelta(hours=4),
        stop_loss=49500.0,
        take_profit=51000.0,
        size=1.0,
        pnl=100.0 if win else -100.0,
        pnl_pct=0.002 if win else -0.002,
        status=TradeStatus.CLOSED,
        pattern_name="double_top",
        score=75.0,
        metadata={"max_adverse_excursion": 20.0},
    )


async def _populate(svc: LearningService, n: int = 10):
    for i in range(n):
        await svc.record_trade(
            _closed_trade(i, win=i % 2 == 0),
            indicators=_INDICATORS,
            variables={"confidence": 0.8, "health": 90.0},
        )


@pytest.mark.asyncio
async def test_record_trade_stores_entry():
    svc = LearningService(repository=MemoryKnowledgeRepository(), mode=LearningMode.ONLINE)
    entry = await svc.record_trade(_closed_trade(0, True), indicators=_INDICATORS)
    assert entry.outcome == TradeOutcome.WIN
    assert entry.instrument == "BTCUSDT"
    assert entry.pattern == "double_top"
    assert entry.risk_reward > 0
    assert len(entry.ml_features) == len(FeatureBuilder().feature_names)
    assert await svc.repository.count() == 1


@pytest.mark.asyncio
async def test_online_learning_incremental():
    svc = LearningService(repository=MemoryKnowledgeRepository(), mode=LearningMode.ONLINE)
    await _populate(svc, n=12)
    assert svc._online.samples_seen == 12
    assert svc._online.is_trained
    pred = svc.predict(_INDICATORS, {"confidence": 0.8, "health": 90.0})
    assert 0.0 <= pred.probability <= 1.0
    assert pred.model_name == "knowledge_online"


@pytest.mark.asyncio
async def test_offline_learning_with_cv():
    svc = LearningService(repository=MemoryKnowledgeRepository(), mode=LearningMode.OFFLINE, min_samples=2)
    await _populate(svc, n=12)
    report = await svc.train_offline(n_splits=3)
    assert report["trained"] is True
    assert report["samples"] == 12
    assert "cross_validation" in report
    assert report["cross_validation"]["n_splits"] == 3
    assert len(report["feature_importance"]) == len(FeatureBuilder().feature_names)


@pytest.mark.asyncio
async def test_predict_falls_back_when_untrained():
    svc = LearningService(repository=MemoryKnowledgeRepository(), mode=LearningMode.OFFLINE)
    pred = svc.predict(_INDICATORS)
    assert pred.probability == 0.5
    assert pred.metadata["source"] == "none"


@pytest.mark.asyncio
async def test_stats_aggregation():
    svc = LearningService(repository=MemoryKnowledgeRepository(), mode=LearningMode.OFFLINE, min_samples=100)
    await _populate(svc, n=10)
    stats = await svc.stats()
    assert stats["total_entries"] == 10
    assert stats["wins"] == 5
    assert stats["losses"] == 5
    assert stats["win_rate"] == 0.5
    assert stats["by_pattern"]["double_top"]["count"] == 10


def test_offline_learner_requires_two_samples():
    learner = OfflineLearner()
    report = learner.train([])
    assert report["trained"] is False


def test_online_learner_proba_without_training():
    learner = OnlineLearner()
    assert learner.predict_proba([1.0] * len(FeatureBuilder().feature_names)) == 0.5


def test_feature_builder_padding():
    builder = FeatureBuilder()
    vec = builder.build({"rsi": 50.0}, {})
    assert len(vec) == len(builder.feature_names)
    assert vec[0] == 50.0
    assert vec[1] == 0.0


@pytest.mark.asyncio
async def test_learning_mode_switch():
    svc = LearningService(repository=MemoryKnowledgeRepository(), mode=LearningMode.OFFLINE)
    assert svc.mode == LearningMode.OFFLINE
    svc.set_mode(LearningMode.ONLINE)
    assert svc.mode == LearningMode.ONLINE
