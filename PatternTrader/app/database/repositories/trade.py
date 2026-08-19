# mypy: ignore-errors
from __future__ import annotations

from typing import Optional

from sqlalchemy import select

from app.backtesting.models import Trade, TradeStatus
from app.database.base import get_async_session
from app.database.models import Trade as TradeORM


class TradeRepository:
    """Persistencia de operaciones (write-through del ExecutionEngine)."""

    async def add(self, trade: Trade) -> None:
        async with get_async_session() as session:
            session.add(self._to_orm(trade))
            await session.flush()

    async def update_closed(self, trade: Trade) -> None:
        async with get_async_session() as session:
            result = await session.execute(select(TradeORM).where(TradeORM.trade_uuid == trade.id))
            orm = result.scalar_one_or_none()
            if orm is None:
                return
            orm.exit_price = trade.exit_price
            orm.exit_time = trade.exit_time
            orm.status = trade.status.value
            orm.pnl = trade.pnl
            orm.pnl_pct = trade.pnl_pct
            orm.metadata_json = trade.metadata

    async def get(self, trade_uuid: str) -> Optional[Trade]:
        async with get_async_session() as session:
            result = await session.execute(
                select(TradeORM).where(TradeORM.trade_uuid == trade_uuid)
            )
            orm = result.scalar_one_or_none()
            return self._to_model(orm) if orm is not None else None

    async def list(
        self,
        status: Optional[TradeStatus] = None,
        symbol: Optional[str] = None,
        limit: int = 200,
    ) -> list[Trade]:
        async with get_async_session() as session:
            stmt = select(TradeORM).order_by(TradeORM.created_at.desc()).limit(limit)
            if status is not None:
                stmt = stmt.where(TradeORM.status == status.value)
            if symbol is not None:
                stmt = stmt.where(TradeORM.symbol == symbol)
            result = await session.execute(stmt)
            return [self._to_model(orm) for orm in result.scalars()]

    @staticmethod
    def _to_orm(trade: Trade) -> TradeORM:
        return TradeORM(
            trade_uuid=trade.id,
            symbol=trade.symbol,
            timeframe=trade.timeframe,
            direction=trade.direction.value,
            entry_price=trade.entry_price,
            entry_time=trade.entry_time,
            exit_price=trade.exit_price,
            exit_time=trade.exit_time,
            stop_loss=trade.stop_loss,
            take_profit=trade.take_profit,
            size=trade.size,
            pnl=trade.pnl,
            pnl_pct=trade.pnl_pct,
            status=trade.status.value,
            pattern_name=trade.pattern_name,
            score=trade.score,
            metadata_json=trade.metadata,
        )

    @staticmethod
    def _to_model(orm: TradeORM) -> Trade:
        return Trade(
            id=orm.trade_uuid,
            symbol=orm.symbol,
            timeframe=orm.timeframe or "",
            direction=orm.direction or "LONG",
            entry_price=orm.entry_price,
            entry_time=orm.entry_time,
            exit_price=orm.exit_price,
            exit_time=orm.exit_time,
            stop_loss=orm.stop_loss,
            take_profit=orm.take_profit,
            size=orm.size or 1.0,
            pnl=orm.pnl or 0.0,
            pnl_pct=orm.pnl_pct or 0.0,
            status=orm.status or "OPEN",
            pattern_name=orm.pattern_name or "",
            score=orm.score or 0.0,
            metadata=orm.metadata_json or {},
        )
