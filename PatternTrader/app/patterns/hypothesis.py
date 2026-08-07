from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.market.candles.models import Candle
from app.patterns.base_pattern import PatternResult
from app.scoring.models import ScoreResult


class PatternHypothesis(BaseModel):
    """Un patrón detectado tratado como hipótesis de mercado.

    El Pattern Engine solo emite hipótesis (qué patrón es y cómo de sano está);
    la decisión de entrar la toman las estrategias.
    """

    pattern: PatternResult
    indicators: dict[str, float] = Field(default_factory=dict)
    score: Optional[ScoreResult] = None
    ml_probability: float = 0.0
    confirmation_score: Optional[float] = None
    candles: Optional[list[Candle]] = None
    market_structure: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def symbol(self) -> str:
        return self.pattern.symbol

    @property
    def timeframe(self) -> str:
        return self.pattern.timeframe

    @property
    def direction(self) -> str:
        return self.pattern.direction

    @property
    def total_score(self) -> float:
        if self.score is not None:
            return self.score.total_score
        return self.pattern.score
