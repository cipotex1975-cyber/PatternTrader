from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class LifecycleState(str, Enum):
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


class LifecycleTransition(BaseModel):
    from_state: LifecycleState
    to_state: LifecycleState
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    reason: str = ""
    metadata: dict = Field(default_factory=dict)


class LifecycleEvent(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    pattern_id: UUID
    symbol: str
    timeframe: str
    pattern_name: str
    current_state: LifecycleState
    transitions: list[LifecycleTransition] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def is_active(self) -> bool:
        return self.current_state not in {
            LifecycleState.CLOSED,
            LifecycleState.INVALIDATED,
            LifecycleState.EXPIRED,
            LifecycleState.CANCELLED,
            LifecycleState.REJECTED,
        }

    @property
    def total_transitions(self) -> int:
        return len(self.transitions)

    def add_transition(
        self,
        to_state: LifecycleState,
        reason: str = "",
        metadata: dict | None = None,
    ) -> LifecycleTransition:
        transition = LifecycleTransition(
            from_state=self.current_state,
            to_state=to_state,
            reason=reason,
            metadata=metadata or {},
        )
        self.transitions.append(transition)
        self.current_state = to_state
        self.updated_at = datetime.utcnow()

        if not self.is_active:
            self.closed_at = datetime.utcnow()

        return transition
