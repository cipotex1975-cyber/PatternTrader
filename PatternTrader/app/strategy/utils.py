from __future__ import annotations

from typing import Optional

from app.patterns.base_pattern import TradeDirection
from app.patterns.hypothesis import PatternHypothesis
from app.strategy.base import StrategySignal


def base_confidence(hypothesis: PatternHypothesis) -> float:
    """Confianza base combinando score y salud del patrón."""
    score = hypothesis.total_score
    health = hypothesis.pattern.health
    return max(0.0, min(1.0, (score / 100.0 + health / 100.0) / 2.0))


def build_signal(
    hypothesis: PatternHypothesis,
    strategy_name: str,
    *,
    size: float,
    confidence: float,
    reasons: list[str],
    metadata: Optional[dict] = None,
) -> Optional[StrategySignal]:
    """Construye una StrategySignal a partir de los niveles del patrón."""
    pattern = hypothesis.pattern
    if (
        pattern.entry_price is None
        or pattern.stop_loss is None
        or pattern.take_profit is None
    ):
        return None

    direction = (
        pattern.direction.value
        if isinstance(pattern.direction, TradeDirection)
        else str(pattern.direction)
    )

    return StrategySignal(
        strategy_name=strategy_name,
        symbol=pattern.symbol,
        timeframe=pattern.timeframe,
        direction=direction,
        entry_price=pattern.entry_price,
        stop_loss=pattern.stop_loss,
        take_profit=pattern.take_profit,
        size=size,
        confidence=confidence,
        reasons=reasons,
        metadata=metadata or {},
    )
