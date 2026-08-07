# mypy: ignore-errors
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import select

from app.database.base import get_async_session
from app.database.models import Log as LogORM


class LogRepository:
    """Escritura de eventos (WARNING/ERROR) a la tabla ``logs``."""

    async def record(
        self,
        level: str,
        message: str,
        source: str = "",
        metadata: Optional[dict] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        async with get_async_session() as session:
            session.add(
                LogORM(
                    level=level,
                    message=message,
                    source=source,
                    metadata_json=metadata or {},
                    timestamp=timestamp or datetime.utcnow(),
                )
            )
            await session.flush()

    async def list(self, level: Optional[str] = None, limit: int = 500) -> list[LogORM]:
        async with get_async_session() as session:
            stmt = (
                select(LogORM)
                .order_by(LogORM.timestamp.desc())
                .limit(limit)
            )
            if level is not None:
                stmt = stmt.where(LogORM.level == level)
            result = await session.execute(stmt)
            return list(result.scalars())
