# mypy: ignore-errors
from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import delete, select, update

from app.database.base import get_async_session
from app.database.models import Signal as SignalORM
from app.signals.models import Signal, SignalPriority, SignalStatus


class SignalRepository:
    """Persistencia de señales (write-through del SignalEngine)."""

    async def add(self, signal: Signal) -> None:
        async with get_async_session() as session:
            session.add(self._to_orm(signal))
            await session.flush()

    async def update_status(
        self,
        signal_id: UUID,
        status: SignalStatus,
        sent_at: Optional[datetime] = None,
    ) -> None:
        async with get_async_session() as session:
            result = await session.execute(
                select(SignalORM).where(SignalORM.signal_uuid == str(signal_id))
            )
            orm = result.scalar_one_or_none()
            if orm is None:
                return
            orm.status = status.value
            if sent_at is not None:
                orm.sent_at = sent_at

    async def mark_expired(self, signal_id: UUID) -> None:
        """Marca la señal como EXPIRED sin borrarla (conserva el historial)."""
        async with get_async_session() as session:
            await session.execute(
                update(SignalORM)
                .where(SignalORM.signal_uuid == str(signal_id))
                .values(status=SignalStatus.EXPIRED.value)
            )

    async def expire_overdue(self) -> int:
        """Expira en lote las señales PENDING cuyo TTL (expires_at) ya venció.

        Devuelve el número de señales actualizadas.
        """
        async with get_async_session() as session:
            result = await session.execute(
                update(SignalORM)
                .where(
                    SignalORM.status == SignalStatus.PENDING.value,
                    SignalORM.expires_at.is_not(None),
                    SignalORM.expires_at < datetime.utcnow(),
                )
                .values(status=SignalStatus.EXPIRED.value)
            )
            return result.rowcount or 0

    async def get(self, signal_uuid: str) -> Optional[Signal]:
        async with get_async_session() as session:
            result = await session.execute(
                select(SignalORM).where(SignalORM.signal_uuid == signal_uuid)
            )
            orm = result.scalar_one_or_none()
            return self._to_model(orm) if orm is not None else None

    async def list(
        self,
        status: Optional[SignalStatus] = None,
        priority: Optional[SignalPriority] = None,
        symbol: Optional[str] = None,
        data_source: Optional[str] = None,
        limit: int = 100,
    ) -> list[Signal]:
        async with get_async_session() as session:
            stmt = select(SignalORM).order_by(SignalORM.created_at.desc()).limit(limit)
            if status is not None:
                stmt = stmt.where(SignalORM.status == status.value)
            if priority is not None:
                stmt = stmt.where(SignalORM.priority == priority.value)
            if symbol is not None:
                stmt = stmt.where(SignalORM.symbol == symbol)
            if data_source is not None:
                stmt = stmt.where(SignalORM.data_source == data_source)
            result = await session.execute(stmt)
            return [self._to_model(orm) for orm in result.scalars()]

    async def delete(self, signal_id: UUID) -> int:
        async with get_async_session() as session:
            result = await session.execute(
                delete(SignalORM).where(SignalORM.signal_uuid == str(signal_id))
            )
            return result.rowcount or 0

    @staticmethod
    def _to_orm(signal: Signal) -> SignalORM:
        return SignalORM(
            signal_uuid=str(signal.id),
            symbol=signal.symbol,
            timeframe=signal.timeframe,
            pattern_name=signal.pattern_name,
            direction=signal.direction,
            priority=signal.priority.value,
            status=signal.status.value,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            risk_reward_ratio=signal.risk_reward_ratio,
            score=signal.score,
            health=signal.health,
            ml_probability=signal.ml_probability,
            reasons=signal.reasons,
            created_at=signal.created_at,
            sent_at=signal.sent_at,
            expires_at=signal.expires_at,
            data_source=signal.data_source,
            metadata_json=signal.metadata,
        )

    @staticmethod
    def _to_model(orm: SignalORM) -> Signal:
        return Signal(
            id=UUID(orm.signal_uuid),
            symbol=orm.symbol,
            timeframe=orm.timeframe,
            pattern_name=orm.pattern_name or "",
            direction=orm.direction or "LONG",
            priority=SignalPriority(orm.priority),
            status=SignalStatus(orm.status),
            entry_price=orm.entry_price or 0.0,
            stop_loss=orm.stop_loss or 0.0,
            take_profit=orm.take_profit or 0.0,
            risk_reward_ratio=orm.risk_reward_ratio or 0.0,
            score=orm.score or 0.0,
            health=orm.health or 0.0,
            ml_probability=orm.ml_probability,
            reasons=orm.reasons or [],
            created_at=orm.created_at,
            sent_at=orm.sent_at,
            expires_at=orm.expires_at,
            data_source=orm.data_source or "live",
            metadata=orm.metadata_json or {},
        )
