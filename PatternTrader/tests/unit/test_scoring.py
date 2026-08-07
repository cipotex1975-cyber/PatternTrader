import pytest
from datetime import datetime, timezone

from app.market.candles.models import Candle, CandleData
from app.patterns.base_pattern import PatternResult, PatternType
from app.scoring.engine import ScoringEngine


def create_test_pattern():
    return PatternResult(
        pattern_name="double_top",
        pattern_type=PatternType.REVERSAL,
        symbol="BTCUSDT",
        timeframe="1h",
        confidence=0.85,
        health=90.0,
        key_levels={"neckline": 50000, "peak1": 52000, "peak2": 51800},
    )


def create_test_candles():
    candles = []
    for i in range(30):
        candles.append(
            Candle(
                symbol="BTCUSDT",
                timeframe="1h",
                data=CandleData(
                    timestamp=datetime.now(timezone.utc),
                    open=50000 + i * 10,
                    high=50100 + i * 10,
                    low=49900 + i * 10,
                    close=50050 + i * 10,
                    volume=1000 + i * 100,
                ),
            )
        )
    return candles


def test_scoring_engine_initialization():
    engine = ScoringEngine()
    assert engine is not None


def test_scoring_calculates_score():
    engine = ScoringEngine()
    pattern = create_test_pattern()
    candles = create_test_candles()
    indicators = {
        "rsi": 65,
        "macd": 100,
        "macd_signal": 90,
        "ema_21": 50100,
        "ema_50": 50000,
        "atr": 200,
        "volume": 1500,
    }

    result = engine.calculate_score(pattern, indicators, candles)
    assert 0 <= result.total_score <= 100
    assert result.grade is not None
    assert 0 <= result.confidence <= 1


def test_scoring_components():
    engine = ScoringEngine()
    pattern = create_test_pattern()
    indicators = {"rsi": 50, "macd": 0, "macd_signal": 0}

    result = engine.calculate_score(pattern, indicators)
    assert len(result.components) > 0
    assert all(c.weight > 0 for c in result.components)


@pytest.mark.asyncio
async def test_scoring_uses_knowledge_model_when_attached():
    from app.learning.models import LearningMode
    from app.learning.repository import MemoryKnowledgeRepository
    from app.learning.service import LearningService

    svc = LearningService(repository=MemoryKnowledgeRepository(), mode=LearningMode.ONLINE)
    for i in range(12):
        win = i % 2 == 0
        ind = {"rsi": 80.0 if win else 20.0, "atr": 100.0}
        await svc.record_trade(
            {
                "symbol": "BTCUSDT",
                "timeframe": "1h",
                "pattern_name": "double_top",
                "direction": "SHORT",
                "entry_price": 50000,
                "exit_price": 49500 if win else 50500,
                "pnl": 500 if win else -500,
            },
            indicators=ind,
        )
    assert svc.is_trained

    engine = ScoringEngine()
    engine.attach_knowledge(svc)

    pattern = create_test_pattern()
    candles = create_test_candles()
    result = engine.calculate_score(pattern, {"rsi": 50}, candles)

    ml = next(c for c in result.components if c.name == "ml_history")
    assert 0 <= ml.score <= 100
    assert 0 <= result.confidence <= 1
