from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field
from uuid import UUID, uuid4

from app.market.candles.models import Candle


class PatternType(str, Enum):
    REVERSAL = "reversal"
    CONTINUATION = "continuation"
    NEUTRAL = "neutral"


class PatternStatus(str, Enum):
    DETECTED = "DETECTED"
    FORMING = "FORMING"
    WAITING_BREAKOUT = "WAITING_BREAKOUT"
    CONFIRMED = "CONFIRMED"
    SIGNAL_SENT = "SIGNAL_SENT"
    OPEN = "OPEN"
    TP_HIT = "TP_HIT"
    SL_HIT = "SL_HIT"
    CLOSED = "CLOSED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class PatternResult(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    pattern_name: str
    pattern_type: PatternType
    symbol: str
    timeframe: str
    status: PatternStatus = PatternStatus.DETECTED
    confidence: float = Field(ge=0.0, le=1.0)
    health: float = Field(ge=0.0, le=100.0, default=100.0)
    score: float = Field(ge=0.0, le=100.0, default=0.0)
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    key_levels: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    max_confirmation_candles: int = 20
    current_candle_count: int = 0

    model_config = {"arbitrary_types_allowed": True}

    @property
    def is_active(self) -> bool:
        return self.status in {
            PatternStatus.DETECTED,
            PatternStatus.FORMING,
            PatternStatus.WAITING_BREAKOUT,
            PatternStatus.CONFIRMED,
        }

    @property
    def candles_remaining(self) -> int:
        return max(0, self.max_confirmation_candles - self.current_candle_count)

    def update_health(self, new_health: float) -> None:
        self.health = max(0.0, min(100.0, new_health))
        self.updated_at = datetime.utcnow()

    def transition(self, new_status: PatternStatus) -> None:
        self.status = new_status
        self.updated_at = datetime.utcnow()


class BasePattern(ABC):
    """Base class for all pattern detectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Pattern name."""
        ...

    @property
    @abstractmethod
    def pattern_type(self) -> PatternType:
        """Pattern type (reversal, continuation, neutral)."""
        ...

    @property
    @abstractmethod
    def max_confirmation_candles(self) -> int:
        """Maximum candles to confirm the pattern."""
        ...

    @abstractmethod
    def detect(
        self,
        candles: list[Candle],
        symbol: str,
        timeframe: str,
    ) -> Optional[PatternResult]:
        """Detect the pattern in the given candles."""
        ...

    @abstractmethod
    def validate(self, pattern: PatternResult, candles: list[Candle]) -> bool:
        """Validate if the pattern is still valid."""
        ...

    @abstractmethod
    def score(self, pattern: PatternResult, indicators: dict[str, float]) -> float:
        """Calculate the pattern score."""
        ...

    def update(self, pattern: PatternResult, candles: list[Candle]) -> PatternResult:
        """Update pattern state with new candles."""
        pattern.current_candle_count += 1
        pattern.updated_at = datetime.utcnow()

        if pattern.current_candle_count >= pattern.max_confirmation_candles:
            pattern.transition(PatternStatus.EXPIRED)

        return pattern

    def invalidate(self, pattern: PatternResult, reason: str = "") -> PatternResult:
        """Invalidate the pattern."""
        pattern.transition(PatternStatus.INVALIDATED)
        pattern.metadata["invalidation_reason"] = reason
        return pattern

    def statistics(self) -> dict[str, Any]:
        """Get pattern statistics."""
        return {
            "name": self.name,
            "type": self.pattern_type.value,
            "max_confirmation_candles": self.max_confirmation_candles,
        }
