from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class LearningMode(str, Enum):
    OFFLINE = "offline"
    ONLINE = "online"


class TradeOutcome(str, Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"


class KnowledgeEntry(BaseModel):
    """Registro de una operación que alimenta la base de conocimiento."""

    id: UUID = Field(default_factory=uuid4)
    instrument: str
    timeframe: str
    pattern: str
    direction: str = "LONG"
    variables: dict[str, Any] = Field(default_factory=dict)
    indicators: dict[str, float] = Field(default_factory=dict)
    outcome: TradeOutcome = TradeOutcome.LOSS
    pnl: float = 0.0
    pnl_pct: float = 0.0
    drawdown: float = 0.0
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_reward: float = 0.0
    duration_seconds: float = 0.0
    score: float = 0.0
    entry_time: datetime = Field(default_factory=datetime.utcnow)
    exit_time: Optional[datetime] = None
    image_path: str = ""
    ml_features: list[float] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def is_win(self) -> int:
        return 1 if self.outcome == TradeOutcome.WIN else 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KnowledgeEntry):
            return NotImplemented
        return self.id == other.id
