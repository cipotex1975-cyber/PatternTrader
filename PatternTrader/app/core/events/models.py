from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EventType(str, Enum):
    CANDLE_RECEIVED = "candle.received"
    CANDLE_UPDATED = "candle.updated"
    PATTERN_DETECTED = "pattern.detected"
    PATTERN_FORMING = "pattern.forming"
    PATTERN_CONFIRMED = "pattern.confirmed"
    PATTERN_INVALIDATED = "pattern.invalidated"
    PATTERN_EXPIRED = "pattern.expired"
    PATTERN_BREAKOUT = "pattern.breakout"
    SIGNAL_CREATED = "signal.created"
    SIGNAL_SENT = "signal.sent"
    SIGNAL_FAILED = "signal.failed"
    TRADE_OPENED = "trade.opened"
    TRADE_CLOSED = "trade.closed"
    TRADE_UPDATED = "trade.updated"
    ML_PREDICTION = "ml.prediction"
    ML_TRAINED = "ml.trained"
    HEALTH_UPDATED = "health.updated"
    LIFECYCLE_TRANSITION = "lifecycle.transition"
    PROVIDER_CONNECTED = "provider.connected"
    PROVIDER_DISCONNECTED = "provider.disconnected"
    SYSTEM_ERROR = "system.error"
    SYSTEM_WARNING = "system.warning"


class Event(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    type: EventType
    source: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}
