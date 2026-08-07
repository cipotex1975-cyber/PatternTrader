# mypy: ignore-errors
from __future__ import annotations

from sqlalchemy import select

from app.database.base import get_async_session
from app.database.models import Asset as AssetORM


class AssetRepository:
    """Resuelve la tabla ``assets`` (FK requerida por ``patterns.asset_id``)."""

    async def get_or_create(self, symbol: str) -> int:
        async with get_async_session() as session:
            result = await session.execute(
                select(AssetORM).where(AssetORM.symbol == symbol)
            )
            asset = result.scalar_one_or_none()
            if asset is None:
                asset = AssetORM(symbol=symbol, is_active=True)
                session.add(asset)
                await session.flush()
            return asset.id

    async def get(self, symbol: str) -> AssetORM | None:
        async with get_async_session() as session:
            result = await session.execute(
                select(AssetORM).where(AssetORM.symbol == symbol)
            )
            return result.scalar_one_or_none()
