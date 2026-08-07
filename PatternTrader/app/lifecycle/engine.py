from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from app.core.events.bus import get_event_bus
from app.core.events.models import Event, EventType
from app.core.logger import get_logger
from app.lifecycle.models import LifecycleEvent, LifecycleState, LifecycleTransition
from app.patterns.base_pattern import PatternResult, PatternStatus

logger = get_logger("LifecycleEngine")


class LifecycleEngine:
    def __init__(self, repository: Optional[Any] = None) -> None:
        self._lifecycles: dict[UUID, LifecycleEvent] = {}
        self._pattern_lifecycle_map: dict[UUID, UUID] = {}
        self._repository = repository
        self._event_bus = get_event_bus()

    async def register(self, pattern: PatternResult) -> LifecycleEvent:
        lifecycle = LifecycleEvent(
            pattern_id=pattern.id,
            symbol=pattern.symbol,
            timeframe=pattern.timeframe,
            pattern_name=pattern.pattern_name,
            current_state=LifecycleState.DETECTED,
        )

        self._lifecycles[lifecycle.id] = lifecycle
        self._pattern_lifecycle_map[pattern.id] = lifecycle.id

        if self._repository is not None:
            await self._repository.register_pattern(pattern, lifecycle)

        await self._event_bus.publish(
            Event(
                type=EventType.LIFECYCLE_TRANSITION,
                source="LifecycleEngine",
                data={
                    "lifecycle_id": str(lifecycle.id),
                    "pattern_id": str(pattern.id),
                    "state": LifecycleState.DETECTED.value,
                },
            )
        )

        logger.info(
            f"Registered lifecycle for {pattern.pattern_name} on "
            f"{pattern.symbol}:{pattern.timeframe}"
        )
        return lifecycle

    async def transition(
        self,
        lifecycle_id: UUID,
        to_state: LifecycleState,
        reason: str = "",
        metadata: dict | None = None,
    ) -> Optional[LifecycleTransition]:
        lifecycle = self._lifecycles.get(lifecycle_id)
        if not lifecycle:
            logger.warning(f"Lifecycle {lifecycle_id} not found")
            return None

        if not lifecycle.is_active:
            logger.warning(f"Lifecycle {lifecycle_id} is not active")
            return None

        if lifecycle.current_state == to_state:
            logger.debug(f"Lifecycle {lifecycle_id} already in state {to_state.value}; skipping")
            return None

        transition = lifecycle.add_transition(to_state, reason, metadata)

        if self._repository is not None:
            await self._repository.update_transition(lifecycle)

        await self._event_bus.publish(
            Event(
                type=EventType.LIFECYCLE_TRANSITION,
                source="LifecycleEngine",
                data={
                    "lifecycle_id": str(lifecycle_id),
                    "pattern_id": str(lifecycle.pattern_id),
                    "from_state": transition.from_state.value,
                    "to_state": transition.to_state.value,
                    "reason": reason,
                },
            )
        )

        logger.info(f"Lifecycle {lifecycle_id}: {transition.from_state.value} -> {to_state.value}")
        return transition

    async def update_pattern_status(
        self,
        pattern: PatternResult,
        new_status: PatternStatus,
        reason: str = "",
    ) -> None:
        lifecycle_id = self._pattern_lifecycle_map.get(pattern.id)
        if not lifecycle_id:
            return

        state_map = {
            PatternStatus.DETECTED: LifecycleState.DETECTED,
            PatternStatus.FORMING: LifecycleState.FORMING,
            PatternStatus.WAITING_BREAKOUT: LifecycleState.WAITING_BREAKOUT,
            PatternStatus.CONFIRMED: LifecycleState.CONFIRMED,
            PatternStatus.SIGNAL_SENT: LifecycleState.SIGNAL_SENT,
            PatternStatus.OPEN: LifecycleState.OPEN,
            PatternStatus.TP_HIT: LifecycleState.TP_HIT,
            PatternStatus.SL_HIT: LifecycleState.SL_HIT,
            PatternStatus.CLOSED: LifecycleState.CLOSED,
            PatternStatus.INVALIDATED: LifecycleState.INVALIDATED,
            PatternStatus.EXPIRED: LifecycleState.EXPIRED,
            PatternStatus.CANCELLED: LifecycleState.CANCELLED,
            PatternStatus.REJECTED: LifecycleState.REJECTED,
        }

        to_state = state_map.get(new_status)
        if to_state:
            await self.transition(lifecycle_id, to_state, reason)

    def get(self, lifecycle_id: UUID) -> Optional[LifecycleEvent]:
        return self._lifecycles.get(lifecycle_id)

    def get_all(self) -> list[LifecycleEvent]:
        return list(self._lifecycles.values())

    def get_by_pattern(self, pattern_id: UUID) -> Optional[LifecycleEvent]:
        lifecycle_id = self._pattern_lifecycle_map.get(pattern_id)
        if lifecycle_id:
            return self._lifecycles.get(lifecycle_id)
        return None

    async def transition_by_pattern(
        self,
        pattern_id: UUID,
        to_state: LifecycleState,
        reason: str = "",
        metadata: dict | None = None,
    ) -> Optional[LifecycleTransition]:
        """Transiciona el lifecycle asociado a un patrón (usado por el motor de
        trades para realimentar OPEN/TP_HIT/SL_HIT/CLOSED)."""
        lifecycle_id = self._pattern_lifecycle_map.get(pattern_id)
        if lifecycle_id is None:
            logger.warning(f"No lifecycle registered for pattern {pattern_id}")
            return None
        return await self.transition(lifecycle_id, to_state, reason, metadata)

    async def rehydrate(
        self,
        entries: list[tuple[PatternResult, LifecycleEvent]],
    ) -> int:
        """Carga en memoria el estado persistido (es idempotente: solo inserta
        lifecycles no presentes). Devuelve el número de entradas cargadas."""
        loaded = 0
        for pattern, lifecycle in entries:
            if lifecycle.id in self._lifecycles:
                continue
            self._lifecycles[lifecycle.id] = lifecycle
            self._pattern_lifecycle_map[lifecycle.pattern_id] = lifecycle.id
            loaded += 1
        if loaded:
            logger.info(f"Rehydrated {loaded} lifecycles from database")
        return loaded

    async def rehydrate_from_db(self) -> int:
        """Rehidrata el estado desde el repositorio persistente (si existe)."""
        if self._repository is None:
            return 0
        try:
            entries = await self._repository.list()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to read lifecycles from database: {e}")
            return 0
        return await self.rehydrate(entries)

    def get_active(self) -> list[LifecycleEvent]:
        return [lc for lc in self._lifecycles.values() if lc.is_active]

    def get_by_symbol(self, symbol: str) -> list[LifecycleEvent]:
        return [lc for lc in self._lifecycles.values() if lc.symbol == symbol]

    def get_by_state(self, state: LifecycleState) -> list[LifecycleEvent]:
        return [lc for lc in self._lifecycles.values() if lc.current_state == state]

    def get_statistics(self) -> dict[str, int]:
        stats: dict[str, int] = {}
        for state in LifecycleState:
            stats[state.value] = len(self.get_by_state(state))
        return stats
