from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from app.backtesting.models import Trade
from app.lifecycle.models import LifecycleEvent
from app.patterns.base_pattern import PatternResult
from app.signals.models import Signal, SignalStatus


class MemoryLifecycleRepository:
    """Variante en memoria del LifecycleRepository (tests de engines sin DB)."""

    def __init__(self) -> None:
        self.patterns: dict[str, PatternResult] = {}
        self.lifecycles: dict[str, dict[str, Any]] = {}

    async def register_pattern(self, pattern: PatternResult, lifecycle: LifecycleEvent) -> None:
        self.patterns[str(pattern.id)] = pattern
        self.lifecycles[str(lifecycle.pattern_id)] = self._snapshot(lifecycle)

    async def update_transition(self, lifecycle: LifecycleEvent) -> None:
        self.lifecycles[str(lifecycle.pattern_id)] = self._snapshot(lifecycle)

    @staticmethod
    def _snapshot(lifecycle: LifecycleEvent) -> dict[str, Any]:
        return {
            "current_state": lifecycle.current_state.value,
            "transitions": [t.model_dump(mode="json") for t in lifecycle.transitions],
            "closed_at": lifecycle.closed_at,
        }


class MemorySignalRepository:
    """Variante en memoria del SignalRepository (tests de engines sin DB)."""

    def __init__(self) -> None:
        self.signals: dict[UUID, Signal] = {}

    async def add(self, signal: Signal) -> None:
        self.signals[signal.id] = signal

    async def update_status(
        self,
        signal_id: UUID,
        status: SignalStatus,
        sent_at: Optional[datetime] = None,
    ) -> None:
        signal = self.signals.get(signal_id)
        if signal is None:
            return
        signal.status = status
        if sent_at is not None:
            signal.sent_at = sent_at


class MemoryTradeRepository:
    """Variante en memoria del TradeRepository (tests de engines sin DB)."""

    def __init__(self) -> None:
        self.trades: dict[str, Trade] = {}

    async def add(self, trade: Trade) -> None:
        self.trades[trade.id] = trade

    async def update_closed(self, trade: Trade) -> None:
        self.trades[trade.id] = trade
