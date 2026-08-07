from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from app.backtesting.models import Trade, TradeDirection, TradeStatus
from app.core.events.bus import get_event_bus
from app.core.events.models import Event, EventType
from app.core.logger import get_logger
from app.execution.models import ExitReason
from app.lifecycle.models import LifecycleState

logger = get_logger("ExecutionEngine")


class ExecutionEngine:
    """Motor de trades: abre posiciones desde las señales enviadas y las cierra
    monitorizando los candles, publicando ``TRADE_OPENED``/``TRADE_CLOSED`` y
    realimentando el lifecycle (OPEN → TP_HIT/SL_HIT → CLOSED)."""

    def __init__(
        self,
        lifecycle: Any = None,
        default_size: float = 1.0,
        max_open_positions: int = 50,
        repository: Optional[Any] = None,
    ) -> None:
        self._lifecycle = lifecycle
        self._default_size = default_size
        self._max_open_positions = max_open_positions
        self._repository = repository
        self._bus = get_event_bus()
        self._open_trades: dict[str, Trade] = {}
        self._closed_trades: dict[str, Trade] = {}
        self._processed_signals: set[str] = set()
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._bus.subscribe(EventType.SIGNAL_SENT, self._on_signal_sent)
        self._bus.subscribe(EventType.CANDLE_UPDATED, self._on_candle)
        self._started = True
        logger.info("ExecutionEngine started (paper trading)")

    async def stop(self) -> None:
        if not self._started:
            return
        self._bus.unsubscribe(EventType.SIGNAL_SENT, self._on_signal_sent)
        self._bus.unsubscribe(EventType.CANDLE_UPDATED, self._on_candle)
        self._started = False
        logger.info("ExecutionEngine stopped")

    @property
    def started(self) -> bool:
        return self._started

    async def _on_signal_sent(self, event: Event) -> None:
        data = event.data or {}
        signal_id = str(data.get("signal_id") or "")
        if signal_id and signal_id in self._processed_signals:
            logger.debug(f"Signal {signal_id} already processed; skipping")
            return

        trade = await self.open_trade(data)
        if trade is None:
            pattern_id = data.get("pattern_id")
            if self._lifecycle is not None and pattern_id:
                try:
                    await self._lifecycle.transition_by_pattern(
                        UUID(str(pattern_id)),
                        LifecycleState.CANCELLED,
                        "invalid execution data: trade could not be opened",
                    )
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Could not cancel lifecycle for {pattern_id}: {e}")
            return

        if trade is not None and signal_id:
            self._processed_signals.add(signal_id)
        logger.info(
            f"Trade opened: {trade.symbol}:{trade.timeframe} "
            f"{trade.direction.value} @ {trade.entry_price}"
        )

    async def open_trade(self, data: dict[str, Any]) -> Optional[Trade]:
        """Abre una posición simulada a partir de una señal enviada."""
        symbol = data.get("symbol")
        entry_price = data.get("entry_price")
        stop_loss = data.get("stop_loss")
        take_profit = data.get("take_profit")

        if not symbol or not entry_price or not stop_loss or not take_profit:
            logger.warning(
                f"Invalid trade payload for {symbol}: missing price levels"
            )
            return None
        if entry_price <= 0 or stop_loss <= 0 or take_profit <= 0:
            logger.warning(f"Invalid price levels for {symbol}")
            return None
        if len(self._open_trades) >= self._max_open_positions:
            logger.warning(f"Max open positions reached ({self._max_open_positions})")
            return None

        trade = Trade(
            id=str(uuid4()),
            symbol=symbol,
            timeframe=str(data.get("timeframe", "")),
            direction=TradeDirection(data.get("direction", "LONG")),
            entry_price=float(entry_price),
            entry_time=datetime.utcnow(),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            size=float(data.get("size") or self._default_size),
            status=TradeStatus.OPEN,
            pattern_name=str(data.get("pattern_name", "")),
            score=float(data.get("score") or 0.0),
            metadata={
                "pattern_id": str(data.get("pattern_id") or ""),
                "signal_id": str(data.get("signal_id") or ""),
                "strategy": str(data.get("strategy") or ""),
                "features": data.get("indicators") or {},
            },
        )

        self._open_trades[trade.id] = trade

        if self._repository is not None:
            await self._repository.add(trade)

        await self._bus.publish(
            Event(
                type=EventType.TRADE_OPENED,
                source="ExecutionEngine",
                data=trade.model_dump(mode="json"),
            )
        )

        await self._transition_lifecycle(
            trade, LifecycleState.OPEN, f"Trade opened @ {trade.entry_price}"
        )

        return trade

    async def _on_candle(self, event: Event) -> None:
        data = event.data or {}
        symbol = data.get("symbol")
        timeframe = data.get("timeframe")
        if not symbol:
            return

        for trade_id in list(self._open_trades.keys()):
            trade = self._open_trades[trade_id]
            if trade.symbol != symbol or (timeframe and trade.timeframe != timeframe):
                continue

            reason = self._check_exit(trade, data.get("high"), data.get("low"))
            if reason is None:
                continue

            exit_price = trade.stop_loss if reason == ExitReason.STOP_LOSS else trade.take_profit
            if exit_price is None:
                logger.warning(f"Missing exit price for trade {trade.id}")
                continue
            await self.close_trade(trade.id, exit_price, reason)

    @staticmethod
    def _check_exit(trade: Trade, high: Any, low: Any) -> Optional[ExitReason]:
        if high is None or low is None:
            return None
        if trade.direction == TradeDirection.LONG:
            if low <= trade.stop_loss:
                return ExitReason.STOP_LOSS
            if high >= trade.take_profit:
                return ExitReason.TAKE_PROFIT
        else:
            if high >= trade.stop_loss:
                return ExitReason.STOP_LOSS
            if low <= trade.take_profit:
                return ExitReason.TAKE_PROFIT
        return None

    async def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        reason: ExitReason = ExitReason.MANUAL,
    ) -> Optional[Trade]:
        """Cierra una posición abierta, publica ``TRADE_CLOSED`` y realimenta el
        lifecycle (TP_HIT/SL_HIT → CLOSED)."""
        trade = self._open_trades.get(trade_id)
        if trade is None:
            logger.warning(f"Trade {trade_id} not open")
            return None

        trade.exit_price = float(exit_price)
        trade.exit_time = datetime.utcnow()
        trade.status = TradeStatus.CLOSED
        trade.metadata["exit_reason"] = reason.value

        if trade.direction == TradeDirection.LONG:
            trade.pnl = (trade.exit_price - trade.entry_price) * trade.size
        else:
            trade.pnl = (trade.entry_price - trade.exit_price) * trade.size
        trade.pnl_pct = (
            trade.pnl / (trade.entry_price * trade.size) if trade.entry_price else 0.0
        )

        self._open_trades.pop(trade_id, None)
        self._closed_trades[trade_id] = trade

        if self._repository is not None:
            await self._repository.update_closed(trade)

        await self._bus.publish(
            Event(
                type=EventType.TRADE_CLOSED,
                source="ExecutionEngine",
                data=trade.model_dump(mode="json"),
            )
        )

        if reason in (ExitReason.TAKE_PROFIT, ExitReason.STOP_LOSS):
            state = (
                LifecycleState.TP_HIT
                if reason == ExitReason.TAKE_PROFIT
                else LifecycleState.SL_HIT
            )
            await self._transition_lifecycle(
                trade, state, f"{reason.value} hit @ {trade.exit_price}"
            )
        await self._transition_lifecycle(
            trade, LifecycleState.CLOSED, f"Trade closed ({reason.value})"
        )

        logger.info(
            f"Trade closed: {trade.symbol} {trade.direction.value} "
            f"pnl={trade.pnl:.2f} ({reason.value})"
        )
        return trade

    async def _transition_lifecycle(
        self, trade: Trade, state: LifecycleState, reason: str
    ) -> None:
        if self._lifecycle is None:
            return
        pattern_id = (trade.metadata or {}).get("pattern_id")
        if not pattern_id:
            return
        try:
            await self._lifecycle.transition_by_pattern(
                UUID(str(pattern_id)), state, reason
            )
        except (ValueError, AttributeError) as e:
            logger.warning(
                f"Could not transition lifecycle for pattern {pattern_id}: {e}"
            )

    def get_open_trades(self) -> list[Trade]:
        return list(self._open_trades.values())

    def get_closed_trades(self) -> list[Trade]:
        return list(self._closed_trades.values())

    def get_trades(self) -> list[Trade]:
        return self.get_open_trades() + self.get_closed_trades()
