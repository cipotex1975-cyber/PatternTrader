# mypy: ignore-errors
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import select

from app.database.base import get_async_session
from app.database.models import MLModel as MLModelORM


class MLModelRepository:
    """Persistencia de registros de modelos ML (no el artefacto en disco)."""

    async def upsert(
        self,
        name: str,
        model_type: str = "",
        version: str = "",
        path: str = "",
        metrics: Optional[dict[str, Any]] = None,
        is_active: bool = False,
        trained_at: Optional[datetime] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        async with get_async_session() as session:
            result = await session.execute(
                select(MLModelORM).where(MLModelORM.name == name)
            )
            orm = result.scalar_one_or_none()
            if orm is None:
                orm = MLModelORM(name=name)
                session.add(orm)
            orm.model_type = model_type
            orm.version = version
            orm.path = path
            orm.metrics = metrics or {}
            orm.is_active = is_active
            if trained_at is not None:
                orm.trained_at = trained_at
            if metadata is not None:
                orm.metadata_json = metadata
            await session.flush()

    async def get(self, name: str) -> Optional[dict[str, Any]]:
        async with get_async_session() as session:
            result = await session.execute(
                select(MLModelORM).where(MLModelORM.name == name)
            )
            orm = result.scalar_one_or_none()
            return self._to_dict(orm) if orm is not None else None

    async def list(self, limit: int = 100) -> list[dict[str, Any]]:
        async with get_async_session() as session:
            stmt = select(MLModelORM).order_by(MLModelORM.created_at.desc()).limit(limit)
            result = await session.execute(stmt)
            return [self._to_dict(orm) for orm in result.scalars()]

    @staticmethod
    def _to_dict(orm: MLModelORM) -> dict[str, Any]:
        return {
            "id": orm.id,
            "name": orm.name,
            "model_type": orm.model_type,
            "version": orm.version,
            "path": orm.path,
            "metrics": orm.metrics or {},
            "is_active": orm.is_active,
            "trained_at": orm.trained_at.isoformat() if orm.trained_at else None,
            "created_at": orm.created_at.isoformat() if orm.created_at else None,
            "metadata": orm.metadata_json or {},
        }
