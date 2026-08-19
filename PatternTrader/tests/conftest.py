from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator

import pytest

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/pattern_trader_test",
)


async def _probe_postgres() -> bool:
    try:
        import asyncpg
    except ImportError:
        return False
    try:
        conn = await asyncpg.connect(TEST_DATABASE_URL.replace("+asyncpg", ""))
        await conn.close()
        return True
    except Exception:
        return False


def _postgres_available() -> bool:
    return asyncio.run(_probe_postgres())


requires_postgres = pytest.mark.skipif(
    not _postgres_available(),
    reason="Postgres no disponible: docker compose up -d (ver INSTALLATION.md)",
)


def _ensure_test_database() -> None:
    asyncio.run(_ensure_test_database_async())


async def _ensure_test_database_async() -> None:
    import asyncpg

    url = TEST_DATABASE_URL.replace("+asyncpg", "")
    try:
        conn = await asyncpg.connect(url)
        await conn.close()
        return
    except Exception:
        pass
    await _create_database()


async def _create_database() -> None:
    import asyncpg

    name = TEST_DATABASE_URL.rsplit("/", 1)[1]
    admin = TEST_DATABASE_URL.replace(f"/{name}", "/postgres")
    conn = await asyncpg.connect(admin)
    try:
        await conn.execute(f'CREATE DATABASE "{name}"')
    except asyncpg.exceptions.DuplicateDatabaseError:
        pass
    finally:
        await conn.close()


@pytest.fixture(autouse=True, scope="session")
def _ensure_test_db():
    try:
        _ensure_test_database()
    except Exception:
        pass


async def _truncate_all_tables() -> None:
    from sqlalchemy import text

    from app.database.base import Base, get_engine

    tables = ", ".join(table.name for table in reversed(Base.metadata.sorted_tables))
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE"))


@pytest.fixture
async def pg_db(tmp_path) -> AsyncIterator[None]:
    from app.core.config.settings import get_settings
    from app.database.base import init_db, reset_engine

    settings = get_settings()
    original_dedup = settings.telegram.dedup_store_path
    settings.telegram.dedup_store_path = str(tmp_path / "telegram_dedup.json")

    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    reset_engine()
    await init_db()
    yield
    await _truncate_all_tables()
    reset_engine()
    settings.telegram.dedup_store_path = original_dedup
