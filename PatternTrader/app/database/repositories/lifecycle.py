# mypy: ignore-errors
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import select

from app.database.base import get_async_session
from app.database.models import Asset as AssetORM
from app.database.models import Lifecycle as LifecycleORM
from app.database.models import Pattern as PatternORM
from app.database.repositories.asset import AssetRepository
from app.lifecycle.models import LifecycleEvent, LifecycleState, LifecycleTransition
from app.patterns.base_pattern import PatternResult, PatternStatus, PatternType, TradeDirection


class LifecycleRepository:
    """Persiste ``patterns`` + ``lifecycles`` (write-through del LifecycleEngine).

    El ``pattern_id`` de la fila ``lifecycles`` es la PK entera del patrón, por
    lo que ambos registros se crean en la misma sesión.
    """

    def __init__(self, asset_repository: AssetRepository | None = None) -> None:
        self._assets = asset_repository or AssetRepository()

    async def register_pattern(
        self, pattern: PatternResult, lifecycle: LifecycleEvent
    ) -> None:
        asset_id = await self._assets.get_or_create(pattern.symbol)
        async with get_async_session() as session:
            pattern_orm = PatternORM(
                pattern_uuid=str(pattern.id),
                asset_id=asset_id,
                timeframe=pattern.timeframe,
                pattern_name=pattern.pattern_name,
                pattern_type=pattern.pattern_type.value,
                confidence=pattern.confidence,
                health=pattern.health,
                score=pattern.score,
                entry_price=pattern.entry_price,
                stop_loss=pattern.stop_loss,
                take_profit=pattern.take_profit,
                risk_reward_ratio=pattern.risk_reward_ratio,
                key_levels=pattern.key_levels or {},
                status=pattern.status.value,
                detected_at=pattern.detected_at,
                updated_at=pattern.updated_at,
                expires_at=pattern.expires_at,
                metadata_json=pattern.metadata or {},
            )
            session.add(pattern_orm)
            await session.flush()

            session.add(
                LifecycleORM(
                    lifecycle_uuid=str(lifecycle.id),
                    pattern_id=pattern_orm.id,
                    current_state=lifecycle.current_state.value,
                    transitions=self._serialize_transitions(lifecycle.transitions),
                    closed_at=lifecycle.closed_at,
                )
            )
            await session.flush()

    async def update_transition(self, lifecycle: LifecycleEvent) -> None:
        async with get_async_session() as session:
            pattern_orm = await self._get_pattern(session, lifecycle.pattern_id)
            if pattern_orm is None:
                return
            result = await session.execute(
                select(LifecycleORM).where(LifecycleORM.pattern_id == pattern_orm.id)
            )
            lifecycle_orm = result.scalar_one_or_none()
            if lifecycle_orm is None:
                return
            lifecycle_orm.current_state = lifecycle.current_state.value
            lifecycle_orm.transitions = self._serialize_transitions(
                lifecycle.transitions
            )
            lifecycle_orm.closed_at = lifecycle.closed_at

    @staticmethod
    async def _get_pattern(session, pattern_id: UUID) -> PatternORM | None:
        result = await session.execute(
            select(PatternORM).where(PatternORM.pattern_uuid == str(pattern_id))
        )
        return result.scalar_one_or_none()

    @staticmethod
    def _serialize_transitions(
        transitions: list[LifecycleTransition],
    ) -> list[dict]:
        return [
            {
                "from_state": t.from_state.value,
                "to_state": t.to_state.value,
                "timestamp": t.timestamp.isoformat(),
                "reason": t.reason,
                "metadata": t.metadata,
            }
            for t in transitions
        ]

    async def list(self, limit: int = 500) -> list[tuple[PatternResult, LifecycleEvent]]:
        """Rehidrata el estado persistido: ``PatternResult`` + ``LifecycleEvent``
        por cada registro de ``patterns`` (join con ``lifecycles`` y ``assets``)."""
        async with get_async_session() as session:
            stmt = (
                select(PatternORM, LifecycleORM, AssetORM)
                .join(LifecycleORM, LifecycleORM.pattern_id == PatternORM.id)
                .join(AssetORM, AssetORM.id == PatternORM.asset_id)
                .order_by(LifecycleORM.updated_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            rows = result.all()
            return [
                (
                    self._pattern_to_model(pattern, asset.symbol),
                    self._lifecycle_to_model(lifecycle, pattern, asset.symbol),
                )
                for pattern, lifecycle, asset in rows
            ]

    @staticmethod
    def _pattern_to_model(pattern: PatternORM, symbol: str) -> PatternResult:
        return PatternResult(
            id=UUID(pattern.pattern_uuid),
            pattern_name=pattern.pattern_name,
            pattern_type=PatternType(pattern.pattern_type or "reversal"),
            symbol=symbol,
            timeframe=pattern.timeframe,
            direction=TradeDirection.LONG,
            status=PatternStatus(pattern.status or "DETECTED"),
            confidence=pattern.confidence or 0.0,
            health=pattern.health or 100.0,
            score=pattern.score or 0.0,
            entry_price=pattern.entry_price,
            stop_loss=pattern.stop_loss,
            take_profit=pattern.take_profit,
            risk_reward_ratio=pattern.risk_reward_ratio,
            key_levels=pattern.key_levels or {},
            metadata=pattern.metadata_json or {},
            detected_at=pattern.detected_at or datetime.utcnow(),
            updated_at=pattern.updated_at or datetime.utcnow(),
            expires_at=pattern.expires_at,
        )

    @staticmethod
    def _lifecycle_to_model(
        lifecycle: LifecycleORM, pattern: PatternORM, symbol: str
    ) -> LifecycleEvent:
        return LifecycleEvent(
            id=UUID(lifecycle.lifecycle_uuid),
            pattern_id=UUID(pattern.pattern_uuid),
            symbol=symbol,
            timeframe=pattern.timeframe,
            pattern_name=pattern.pattern_name,
            current_state=LifecycleState(lifecycle.current_state),
            transitions=LifecycleRepository._deserialize_transitions(
                lifecycle.transitions or []
            ),
            created_at=lifecycle.created_at or datetime.utcnow(),
            updated_at=lifecycle.updated_at or datetime.utcnow(),
            closed_at=lifecycle.closed_at,
        )

    @staticmethod
    def _deserialize_transitions(raw: list[dict]) -> list[LifecycleTransition]:
        transitions: list[LifecycleTransition] = []
        for item in raw:
            try:
                transitions.append(
                    LifecycleTransition(
                        from_state=LifecycleState(item["from_state"]),
                        to_state=LifecycleState(item["to_state"]),
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        reason=item.get("reason", ""),
                        metadata=item.get("metadata", {}),
                    )
                )
            except (KeyError, ValueError):
                continue
        return transitions
