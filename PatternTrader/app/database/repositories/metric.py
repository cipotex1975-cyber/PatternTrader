# mypy: ignore-errors
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import select

from app.database.base import get_async_session
from app.database.models import Metric as MetricORM


class MetricRepository:
    """Registro de métricas puntuales del sistema."""

    async def record(
        self,
        name: str,
        value: float,
        tags: Optional[dict] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        async with get_async_session() as session:
            session.add(
                MetricORM(
                    name=name,
                    value=value,
                    tags=tags or {},
                    timestamp=timestamp or datetime.utcnow(),
                )
            )
            await session.flush()

    async def list(self, name: Optional[str] = None, limit: int = 1000) -> list[MetricORM]:
        async with get_async_session() as session:
            stmt = select(MetricORM).order_by(MetricORM.timestamp.desc()).limit(limit)
            if name is not None:
                stmt = stmt.where(MetricORM.name == name)
            result = await session.execute(stmt)
            return list(result.scalars())
