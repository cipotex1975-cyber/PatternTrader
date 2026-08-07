from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine

from app.core.logger import get_logger

logger = get_logger("Scheduler")


class Scheduler:
    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task] = {}
        self._running = False

    async def add_interval(
        self,
        name: str,
        func: Callable[..., Coroutine[Any, Any, None]],
        interval_seconds: float,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if name in self._tasks:
            logger.warning(f"Task {name} already exists")
            return

        async def _run_periodically() -> None:
            while self._running:
                try:
                    await func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in task {name}: {e}")
                await asyncio.sleep(interval_seconds)

        task = asyncio.create_task(_run_periodically())
        self._tasks[name] = task
        logger.info(f"Added interval task: {name} (every {interval_seconds}s)")

    async def add_cron(
        self,
        name: str,
        func: Callable[..., Coroutine[Any, Any, None]],
        hour: int = 0,
        minute: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if name in self._tasks:
            logger.warning(f"Task {name} already exists")
            return

        async def _run_daily() -> None:
            while self._running:
                now = datetime.utcnow()
                target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if target <= now:
                    target += timedelta(days=1)
                await asyncio.sleep((target - now).total_seconds())
                try:
                    await func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in task {name}: {e}")

        task = asyncio.create_task(_run_daily())
        self._tasks[name] = task
        logger.info(f"Added cron task: {name} (at {hour:02d}:{minute:02d})")

    async def remove(self, name: str) -> None:
        if name in self._tasks:
            self._tasks[name].cancel()
            try:
                await self._tasks[name]
            except asyncio.CancelledError:
                pass
            del self._tasks[name]
            logger.info(f"Removed task: {name}")

    async def start(self) -> None:
        self._running = True
        logger.info("Scheduler started")

    async def stop(self) -> None:
        self._running = False
        for name in list(self._tasks.keys()):
            await self.remove(name)
        logger.info("Scheduler stopped")

    def get_tasks(self) -> list[str]:
        return list(self._tasks.keys())
