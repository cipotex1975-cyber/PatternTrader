from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import delete, select

from app.core.logger import get_logger
from app.database.base import get_async_session
from app.database.models import KnowledgeEntry as KnowledgeEntryORM
from app.learning.models import KnowledgeEntry, TradeOutcome

logger = get_logger("KnowledgeRepository")


class KnowledgeRepository:
    """Persistencia de la base de conocimiento (SQLAlchemy async)."""

    async def add(self, entry: KnowledgeEntry) -> KnowledgeEntry:
        async with get_async_session() as session:
            orm = self._to_orm(entry)
            session.add(orm)
            await session.flush()
        return entry

    async def add_many(self, entries: list[KnowledgeEntry]) -> int:
        async with get_async_session() as session:
            session.add_all([self._to_orm(e) for e in entries])
            await session.flush()
        return len(entries)

    async def list(
        self,
        instrument: Optional[str] = None,
        timeframe: Optional[str] = None,
        pattern: Optional[str] = None,
        outcome: Optional[TradeOutcome] = None,
        limit: int = 1000,
    ) -> list[KnowledgeEntry]:
        async with get_async_session() as session:
            stmt = (
                select(KnowledgeEntryORM)
                .order_by(KnowledgeEntryORM.created_at.desc())
                .limit(limit)
            )
            if instrument:
                stmt = stmt.where(KnowledgeEntryORM.instrument == instrument)
            if timeframe:
                stmt = stmt.where(KnowledgeEntryORM.timeframe == timeframe)
            if pattern:
                stmt = stmt.where(KnowledgeEntryORM.pattern == pattern)
            if outcome:
                stmt = stmt.where(KnowledgeEntryORM.outcome == outcome.value)
            result = await session.execute(stmt)
            return [self._to_model(r) for r in result.scalars()]

    async def get_all(self, limit: int = 10000) -> list[KnowledgeEntry]:
        return await self.list(limit=limit)

    async def count(self) -> int:
        async with get_async_session() as session:
            result = await session.execute(select(KnowledgeEntryORM))
            return len(result.scalars().all())

    async def clear(self) -> None:
        async with get_async_session() as session:
            await session.execute(delete(KnowledgeEntryORM))
            await session.commit()

    @staticmethod
    def _to_orm(entry: KnowledgeEntry) -> KnowledgeEntryORM:
        return KnowledgeEntryORM(
            entry_uuid=str(entry.id),
            instrument=entry.instrument,
            timeframe=entry.timeframe,
            pattern=entry.pattern,
            direction=entry.direction,
            variables=entry.variables,
            indicators=entry.indicators,
            outcome=entry.outcome.value,
            pnl=entry.pnl,
            pnl_pct=entry.pnl_pct,
            drawdown=entry.drawdown,
            take_profit=entry.take_profit,
            stop_loss=entry.stop_loss,
            risk_reward=entry.risk_reward,
            duration_seconds=entry.duration_seconds,
            score=entry.score,
            entry_time=entry.entry_time,
            exit_time=entry.exit_time,
            image_path=entry.image_path,
            ml_features=entry.ml_features,
        )

    @staticmethod
    def _to_model(orm: KnowledgeEntryORM) -> KnowledgeEntry:
        return KnowledgeEntry(
            id=orm.entry_uuid,
            instrument=orm.instrument,
            timeframe=orm.timeframe,
            pattern=orm.pattern,
            direction=orm.direction or "LONG",
            variables=orm.variables or {},
            indicators=orm.indicators or {},
            outcome=TradeOutcome(orm.outcome),
            pnl=orm.pnl or 0.0,
            pnl_pct=orm.pnl_pct or 0.0,
            drawdown=orm.drawdown or 0.0,
            take_profit=orm.take_profit,
            stop_loss=orm.stop_loss,
            risk_reward=orm.risk_reward or 0.0,
            duration_seconds=orm.duration_seconds or 0.0,
            score=orm.score or 0.0,
            entry_time=orm.entry_time,
            exit_time=orm.exit_time,
            image_path=orm.image_path or "",
            ml_features=orm.ml_features or [],
            created_at=orm.created_at,
        )


class MemoryKnowledgeRepository:
    """Versión en memoria para tests y entornos sin base de datos."""

    def __init__(self) -> None:
        self._entries: dict[Any, KnowledgeEntry] = {}

    async def add(self, entry: KnowledgeEntry) -> KnowledgeEntry:
        self._entries[entry.id] = entry
        return entry

    async def add_many(self, entries: list[KnowledgeEntry]) -> int:
        for entry in entries:
            self._entries[entry.id] = entry
        return len(entries)

    async def list(
        self,
        instrument: Optional[str] = None,
        timeframe: Optional[str] = None,
        pattern: Optional[str] = None,
        outcome: Optional[TradeOutcome] = None,
        limit: int = 1000,
    ) -> list[KnowledgeEntry]:
        result = list(self._entries.values())
        if instrument:
            result = [e for e in result if e.instrument == instrument]
        if timeframe:
            result = [e for e in result if e.timeframe == timeframe]
        if pattern:
            result = [e for e in result if e.pattern == pattern]
        if outcome:
            result = [e for e in result if e.outcome == outcome]
        result.sort(key=lambda e: e.created_at, reverse=True)
        return result[:limit]

    async def get_all(self, limit: int = 10000) -> list[KnowledgeEntry]:
        return await self.list(limit=limit)

    async def count(self) -> int:
        return len(self._entries)

    async def clear(self) -> None:
        self._entries.clear()
