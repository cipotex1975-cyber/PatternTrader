from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class SignalPriority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SignalStatus(str, Enum):
    PENDING = "PENDING"
    SENT = "SENT"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


class Signal(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    symbol: str
    timeframe: str
    pattern_name: str
    direction: str
    priority: SignalPriority
    status: SignalStatus = SignalStatus.PENDING
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    score: float
    health: float
    ml_probability: Optional[float] = None
    reasons: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    sent_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: dict = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def is_expired(self) -> bool:
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False

    @property
    def priority_score(self) -> float:
        priority_map = {
            SignalPriority.LOW: 25,
            SignalPriority.MEDIUM: 50,
            SignalPriority.HIGH: 75,
            SignalPriority.CRITICAL: 100,
        }
        return priority_map.get(self.priority, 0)

    def mark_sent(self) -> None:
        self.status = SignalStatus.SENT
        self.sent_at = datetime.utcnow()

    def mark_delivered(self) -> None:
        self.status = SignalStatus.DELIVERED

    def mark_failed(self, reason: str = "") -> None:
        self.status = SignalStatus.FAILED
        self.metadata["failure_reason"] = reason
