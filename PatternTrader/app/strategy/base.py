from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class StrategySignal(BaseModel):
    symbol: str
    timeframe: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseStrategy(ABC):
    """Base class for all trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Strategy description."""
        ...

    @abstractmethod
    async def evaluate(
        self,
        symbol: str,
        timeframe: str,
        data: dict[str, Any],
    ) -> Optional[StrategySignal]:
        """Evaluate market conditions and generate a signal."""
        ...

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Get strategy parameters."""
        ...

    @abstractmethod
    def set_parameters(self, parameters: dict[str, Any]) -> None:
        """Set strategy parameters."""
        ...

    def should_exit(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> tuple[bool, str]:
        """Check if a position should be closed."""
        if current_price <= stop_loss:
            return True, "STOP_LOSS"
        if current_price >= take_profit:
            return True, "TAKE_PROFIT"
        return False, ""
