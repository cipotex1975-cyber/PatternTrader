from __future__ import annotations

import pytest

from app.patterns.base_pattern import PatternResult, PatternType, TradeDirection
from app.patterns.hypothesis import PatternHypothesis
from app.scoring.models import ScoreResult
from app.strategy.base import StrategyDecision
from app.strategy.engine import StrategyEngine
from app.strategy.evaluator import compare_strategies, run_strategy_backtest
from app.strategy.factory import StrategyFactory
from app.strategy.registry import StrategyRegistry


def build_hypothesis(
    direction: TradeDirection = TradeDirection.LONG,
    pattern_type: PatternType = PatternType.CONTINUATION,
    score: float = 80.0,
    health: float = 70.0,
    indicators: dict[str, float] | None = None,
    levels: dict[str, float] | None = None,
    market_structure: dict | None = None,
) -> PatternHypothesis:
    pattern = PatternResult(
        pattern_name="test_pattern",
        pattern_type=pattern_type,
        symbol="BTCUSDT",
        timeframe="1h",
        direction=direction,
        confidence=0.8,
        health=health,
        score=score,
        entry_price=100.0,
        stop_loss=98.0,
        take_profit=105.0,
        risk_reward_ratio=2.5,
        key_levels=levels or {},
    )
    score_result = ScoreResult(total_score=score, grade="B", confidence=0.8)
    return PatternHypothesis(
        pattern=pattern,
        indicators=indicators or {},
        score=score_result,
        market_structure=market_structure or {},
    )


def test_registry_registers_all_strategies():
    names = set(StrategyRegistry.get_all().keys())
    assert {"trend_follow", "breakout", "contrarian"} <= names


def test_factory_creates_strategy_with_parameters():
    strategy = StrategyFactory.create("trend_follow", {"default_size": 2.0})
    assert strategy.get_parameters()["default_size"] == 2.0
    assert strategy.name == "trend_follow"


def test_factory_unknown_strategy_raises():
    with pytest.raises(ValueError):
        StrategyFactory.create("unknown_strategy")


def test_trend_follow_enters_long():
    strategy = StrategyFactory.create("trend_follow")
    hypothesis = build_hypothesis(
        direction=TradeDirection.LONG,
        indicators={"ema_9": 101.0, "ema_21": 99.0, "momentum": 2.0},
    )
    decision = strategy.evaluate(hypothesis)
    assert decision.action == "ENTER"
    assert decision.signal is not None
    assert decision.signal.direction == "LONG"
    assert decision.signal.entry_price == 100.0
    assert decision.signal.strategy_name == "trend_follow"


def test_trend_follow_rejects_against_trend():
    strategy = StrategyFactory.create("trend_follow")
    hypothesis = build_hypothesis(
        direction=TradeDirection.LONG,
        indicators={"ema_9": 99.0, "ema_21": 101.0, "momentum": 2.0},
    )
    decision = strategy.evaluate(hypothesis)
    assert decision.action == "NO_TRADE"
    assert decision.signal is None


def test_trend_follow_rejects_negative_momentum_for_long():
    strategy = StrategyFactory.create("trend_follow")
    hypothesis = build_hypothesis(
        direction=TradeDirection.LONG,
        indicators={"ema_9": 101.0, "ema_21": 99.0, "momentum": -1.0},
    )
    decision = strategy.evaluate(hypothesis)
    assert decision.action == "NO_TRADE"


def test_breakout_enters_long():
    strategy = StrategyFactory.create("breakout")
    hypothesis = build_hypothesis(
        direction=TradeDirection.LONG,
        indicators={"rsi": 55.0, "momentum": 1.5},
    )
    decision = strategy.evaluate(hypothesis)
    assert decision.action == "ENTER"
    assert decision.signal is not None


def test_breakout_rejects_overbought_rsi_for_long():
    strategy = StrategyFactory.create("breakout")
    hypothesis = build_hypothesis(
        direction=TradeDirection.LONG,
        indicators={"rsi": 80.0, "momentum": 1.5},
    )
    decision = strategy.evaluate(hypothesis)
    assert decision.action == "NO_TRADE"


def test_breakout_enters_short():
    strategy = StrategyFactory.create("breakout")
    hypothesis = build_hypothesis(
        direction=TradeDirection.SHORT,
        indicators={"rsi": 50.0, "momentum": -1.5},
    )
    decision = strategy.evaluate(hypothesis)
    assert decision.action == "ENTER"
    assert decision.signal is not None
    assert decision.signal.direction == "SHORT"


def test_contrarian_enters_oversold_long():
    strategy = StrategyFactory.create("contrarian")
    hypothesis = build_hypothesis(
        direction=TradeDirection.LONG,
        pattern_type=PatternType.REVERSAL,
        indicators={"rsi": 28.0, "momentum": -1.0, "ema_9": 98.0, "ema_21": 100.0},
    )
    decision = strategy.evaluate(hypothesis)
    assert decision.action == "ENTER"


def test_contrarian_rejects_continuation():
    strategy = StrategyFactory.create("contrarian")
    hypothesis = build_hypothesis(
        direction=TradeDirection.LONG,
        pattern_type=PatternType.CONTINUATION,
        indicators={"rsi": 28.0, "momentum": -1.0, "ema_9": 98.0, "ema_21": 100.0},
    )
    decision = strategy.evaluate(hypothesis)
    assert decision.action == "NO_TRADE"


def test_engine_picks_contrarian_for_reversal():
    engine = StrategyEngine()
    assert "contrarian" in engine.strategy_names
    hypothesis = build_hypothesis(
        direction=TradeDirection.LONG,
        pattern_type=PatternType.REVERSAL,
        indicators={"rsi": 28.0, "momentum": -1.0, "ema_9": 98.0, "ema_21": 100.0},
    )
    result = engine.evaluate(hypothesis)
    assert result.has_entry
    assert result.best is not None
    assert result.best.strategy_name == "contrarian"


def test_engine_no_entry():
    engine = StrategyEngine()
    hypothesis = build_hypothesis(
        direction=TradeDirection.LONG,
        pattern_type=PatternType.CONTINUATION,
        indicators={"ema_9": 99.0, "ema_21": 101.0, "momentum": -1.0, "rsi": 80.0},
    )
    result = engine.evaluate(hypothesis)
    assert not result.has_entry
    assert result.best is None


def test_run_strategy_backtest_win_rate():
    h1 = build_hypothesis(
        direction=TradeDirection.LONG,
        indicators={"ema_9": 101.0, "ema_21": 99.0, "momentum": 2.0},
        market_structure={"outcome": {"pnl_pct": 3.5}},
    )
    h2 = build_hypothesis(
        direction=TradeDirection.LONG,
        indicators={"ema_9": 102.0, "ema_21": 98.0, "momentum": 2.5},
        market_structure={"outcome": {"pnl_pct": -2.0}},
    )
    result = run_strategy_backtest([h1, h2], "trend_follow")
    assert result["entries"] == 2
    assert result["directions"]["LONG"] == 2
    assert result["win_rate"] == 0.5


def test_compare_strategies_covers_all():
    hypothesis = build_hypothesis(
        direction=TradeDirection.LONG,
        indicators={"ema_9": 101.0, "ema_21": 99.0, "momentum": 2.0, "rsi": 55.0},
    )
    results = compare_strategies([hypothesis])
    names = {r["strategy"] for r in results}
    assert {"trend_follow", "breakout"} <= names


def test_strategy_decision_no_trade_shape():
    strategy = StrategyFactory.create("breakout")
    hypothesis = build_hypothesis(
        direction=TradeDirection.LONG,
        indicators={"rsi": 90.0, "momentum": 1.0},
    )
    decision = strategy.evaluate(hypothesis)
    assert isinstance(decision, StrategyDecision)
    assert decision.is_entry is False
    assert decision.reasons
