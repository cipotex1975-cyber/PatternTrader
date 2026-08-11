# mypy: ignore-errors
from __future__ import annotations

import math
from typing import Any, Optional

from sqlalchemy import select

from app.backtesting.models import BacktestConfig, BacktestMetrics, BacktestResult, Trade
from app.database.base import get_async_session
from app.database.models import Backtest as BacktestORM


def _json_safe(value: Any) -> Any:
    """Convierte floats no finitos a None (Postgres JSON no admite Infinity)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


class BacktestRepository:
    """Persistencia de resultados de backtest.

    ``add`` devuelve la PK entera que los routers usan como ``id``; las
    lecturas devuelven un dict ``{"id", "name", "result"}``.
    """

    async def add(self, result: BacktestResult, name: str = "") -> int:
        async with get_async_session() as session:
            orm = self._to_orm(result, name)
            session.add(orm)
            await session.flush()
            return orm.id

    async def get(self, backtest_id: int) -> Optional[dict[str, Any]]:
        async with get_async_session() as session:
            result = await session.execute(
                select(BacktestORM).where(BacktestORM.id == backtest_id)
            )
            orm = result.scalar_one_or_none()
            return self._to_dict(orm) if orm is not None else None

    async def list(self, limit: int = 100) -> list[dict[str, Any]]:
        async with get_async_session() as session:
            stmt = select(BacktestORM).order_by(BacktestORM.created_at.desc()).limit(limit)
            result = await session.execute(stmt)
            return [self._to_dict(orm) for orm in result.scalars()]

    @staticmethod
    def _to_dict(orm: BacktestORM) -> dict[str, Any]:
        return {
            "id": orm.id,
            "name": orm.name or "",
            "result": BacktestRepository._to_model(orm),
        }

    @staticmethod
    def _to_orm(result: BacktestResult, name: str) -> BacktestORM:
        m = result.metrics
        return BacktestORM(
            name=name,
            config=_json_safe(result.config.model_dump(mode="json")),
            metrics=_json_safe(m.model_dump(mode="json")),
            trades=_json_safe([t.model_dump(mode="json") for t in result.trades]),
            equity_curve=_json_safe(result.equity_curve),
            trades_count=len(result.trades),
            win_rate=m.win_rate,
            profit_factor=m.profit_factor,
            sharpe_ratio=m.sharpe_ratio,
            max_drawdown=m.max_drawdown,
            total_pnl=m.total_pnl,
            start_date=result.start_date,
            end_date=result.end_date,
            initial_capital=result.initial_capital,
            final_capital=result.final_capital,
            metadata_json=_json_safe(result.metadata),
        )

    @staticmethod
    def _to_model(orm: BacktestORM) -> BacktestResult:
        config = BacktestConfig(**(orm.config or {}))
        metrics = BacktestMetrics(**(orm.metrics or {}))
        return BacktestResult(
            config=config,
            metrics=metrics,
            trades=BacktestRepository._load_trades(orm.trades or []),
            equity_curve=orm.equity_curve or [],
            start_date=orm.start_date,
            end_date=orm.end_date,
            initial_capital=orm.initial_capital,
            final_capital=orm.final_capital,
            metadata=orm.metadata_json or {},
        )

    @staticmethod
    def _load_trades(raw: list[dict]) -> list[Trade]:
        return [Trade(**t) for t in raw]
