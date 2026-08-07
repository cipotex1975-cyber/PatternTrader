from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.patterns.hypothesis import PatternHypothesis


class StrategySignal(BaseModel):
    """Señal de trading producida por una estrategia a partir de una hipótesis."""

    strategy_name: str
    symbol: str
    timeframe: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class StrategyDecision(BaseModel):
    """Decisión de una estrategia sobre una hipótesis.

    `action` puede ser "ENTER" o "NO_TRADE". Es la salida del paso 2 del
    pipeline: la estrategia decide si la hipótesis merece una entrada.
    """

    strategy_name: str
    action: str = Field(pattern="^(ENTER|NO_TRADE)$")
    signal: Optional[StrategySignal] = None
    reasons: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_entry(self) -> bool:
        return self.action == "ENTER"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class BaseStrategy(ABC):
    """Base class for all trading strategies.

    Las estrategias son consumidoras de hipótesis: reciben un
    `PatternHypothesis` y devuelven una `StrategyDecision`. Son el paso 2 del
    pipeline (qué hacer con el patrón), independientes del patrón en sí.
    """

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

    @property
    def version(self) -> str:
        return "1.0"

    @abstractmethod
    def evaluate(self, hypothesis: PatternHypothesis) -> StrategyDecision:
        """Evaluate a pattern hypothesis and decide whether to enter."""
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

    def _no_trade(self, *reasons: str) -> StrategyDecision:
        return StrategyDecision(
            strategy_name=self.name,
            action="NO_TRADE",
            reasons=list(reasons),
        )
