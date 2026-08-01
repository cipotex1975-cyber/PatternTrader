from __future__ import annotations

from typing import Any, Callable, Coroutine

from app.core.logger import get_logger

logger = get_logger("WebSocketManager")


class WebSocketManager:
    def __init__(self) -> None:
        self._connections: dict[str, Any] = {}
        self._handlers: dict[str, list[Callable[..., Coroutine[Any, Any, None]]]] = {}
        self._running = False

    async def connect(self, url: str, name: str = "default") -> None:
        logger.info(f"Connecting to WebSocket: {url}")
        self._connections[name] = url
        self._running = True

    async def disconnect(self, name: str = "default") -> None:
        if name in self._connections:
            del self._connections[name]
            logger.info(f"Disconnected WebSocket: {name}")

    def subscribe(self, event: str, handler: Callable[..., Coroutine[Any, Any, None]]) -> None:
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)

    async def broadcast(self, event: str, data: Any) -> None:
        handlers = self._handlers.get(event, [])
        for handler in handlers:
            try:
                await handler(data)
            except Exception as e:
                logger.error(f"Error in WebSocket handler: {e}")

    async def start(self) -> None:
        self._running = True
        logger.info("WebSocket manager started")

    async def stop(self) -> None:
        self._running = False
        for name in list(self._connections.keys()):
            await self.disconnect(name)
        logger.info("WebSocket manager stopped")

    def get_connections(self) -> list[str]:
        return list(self._connections.keys())
