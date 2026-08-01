from __future__ import annotations

from app.core.config.settings import get_settings
from app.core.events.bus import get_event_bus
from app.core.events.models import Event, EventType
from app.core.logger import get_logger
from app.signals.models import Signal, SignalPriority

logger = get_logger("TelegramNotifier")


class TelegramNotifier:
    def __init__(self) -> None:
        settings = get_settings()
        self._config = settings.telegram
        self._bot_token = self._config.bot_token
        self._chat_id = self._config.chat_id
        self._enabled = self._config.enabled
        self._event_bus = get_event_bus()

    async def initialize(self) -> None:
        if self._enabled:
            self._event_bus.subscribe(EventType.SIGNAL_CREATED, self._on_signal_created)
            logger.info("Telegram notifier initialized")

    async def send_signal(self, signal: Signal) -> bool:
        if not self._enabled:
            logger.debug("Telegram notifier is disabled")
            return False

        message = self._format_signal_message(signal)

        try:
            await self._send_message(message)
            signal.mark_sent()
            logger.info(f"Signal sent to Telegram: {signal.symbol} {signal.pattern_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to send signal to Telegram: {e}")
            signal.mark_failed(str(e))
            return False

    async def send_message(self, message: str) -> bool:
        if not self._enabled:
            return False

        try:
            await self._send_message(message)
            return True
        except Exception as e:
            logger.error(f"Failed to send message to Telegram: {e}")
            return False

    def _format_signal_message(self, signal: Signal) -> str:
        priority_emoji = {
            SignalPriority.LOW: "🟢",
            SignalPriority.MEDIUM: "🟡",
            SignalPriority.HIGH: "🟠",
            SignalPriority.CRITICAL: "🔴",
        }

        emoji = priority_emoji.get(signal.priority, "⚪")
        direction = "📈 LONG" if signal.direction == "LONG" else "📉 SHORT"

        message = f"""
🚨 **Nueva Señal**

{emoji} **{signal.symbol}**

{direction}

**Patrón:** {signal.pattern_name}
**Score:** {signal.score:.1f}
**Health:** {signal.health:.1f}%
**Probabilidad IA:** {signal.ml_probability:.0%}

**Entrada:** {signal.entry_price:,.2f}
**Stop Loss:** {signal.stop_loss:,.2f}
**Take Profit:** {signal.take_profit:,.2f}
**R/R:** {signal.risk_reward_ratio:.2f}

**Motivo:**
"""
        for reason in signal.reasons[:5]:
            message += f"✔ {reason}\n"

        return message.strip()

    async def _send_message(self, message: str) -> None:
        import httpx

        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()

    async def _on_signal_created(self, event: Event) -> None:
        logger.info(f"Signal created event received: {event.data}")
