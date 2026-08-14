from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Callable, Coroutine

from app.core.events.models import Event, EventType
from app.core.logger import get_logger

logger = get_logger("EventBus")


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Callable[..., Coroutine[Any, Any, None]]]] = (
            defaultdict(list)
        )
        self._queue: asyncio.Queue[Event] | None = None
        self._running = False
        self._task: asyncio.Task | None = None

    def subscribe(
        self, event_type: EventType, handler: Callable[..., Coroutine[Any, Any, None]]
    ) -> None:
        self._handlers[event_type].append(handler)
        logger.debug(f"Handler {handler.__name__} subscribed to {event_type.value}")

    def unsubscribe(
        self, event_type: EventType, handler: Callable[..., Coroutine[Any, Any, None]]
    ) -> None:
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
            logger.debug(f"Handler {handler.__name__} unsubscribed from {event_type.value}")

    async def publish(self, event: Event) -> None:
        if self._queue is None:
            self._queue = asyncio.Queue()
        await self._queue.put(event)
        logger.debug(f"Event {event.type.value} published from {event.source}")

    async def _process_events(self) -> None:
        queue = self._queue
        while self._running:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
                handlers = self._handlers.get(event.type, [])
                for handler in handlers:
                    try:
                        await handler(event)
                    except Exception as e:
                        logger.error(
                            f"Error in handler {handler.__name__} for event {event.type.value}: {e}"
                        )
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")

    async def start(self) -> None:
        if not self._running:
            self._queue = asyncio.Queue()
            self._running = True
            self._task = asyncio.create_task(self._process_events())
            logger.info("Event bus started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._queue = None
        logger.info("Event bus stopped")


_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus
