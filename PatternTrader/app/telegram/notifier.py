from __future__ import annotations

import asyncio
from io import BytesIO
from typing import Any, Optional

from app.core.config.settings import get_settings
from app.core.logger import get_logger
from app.market.candles.models import Candle
from app.patterns.base_pattern import PatternResult
from app.signals.models import Signal, SignalPriority
from app.visualization.charts import ChartGenerator

logger = get_logger("TelegramNotifier")


class TelegramNotifier:
    def __init__(self) -> None:
        settings = get_settings()
        self._config = settings.telegram
        self._bot_token = self._config.bot_token
        self._chat_id = self._config.chat_id
        self._enabled = self._config.enabled
        self._max_retries = self._config.max_retries
        self._retry_backoff = self._config.retry_backoff_seconds
        self._timeout = self._config.timeout_seconds
        self._send_image = self._config.send_image
        self._chart_generator = ChartGenerator()

    async def initialize(self) -> None:
        if self._enabled:
            logger.info("Telegram notifier initialized")

    async def send_signal(
        self,
        signal: Signal,
        candles: Optional[list[Candle]] = None,
        pattern: Optional[PatternResult] = None,
    ) -> bool:
        if not self._enabled:
            logger.debug("Telegram notifier is disabled")
            return False

        message = self._format_signal_message(signal)

        try:
            if self._send_image and candles:
                try:
                    await self._send_photo(message, candles, signal, pattern)
                    logger.info(
                        f"Signal with chart sent to Telegram: "
                        f"{signal.symbol} {signal.pattern_name}"
                    )
                    return True
                except Exception as e:
                    logger.warning(f"Failed to send signal image, falling back to text: {e}")
            await self._send_message(message)
            logger.info(f"Signal sent to Telegram: {signal.symbol} {signal.pattern_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to send signal to Telegram: {e}")
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
        timestamp = signal.created_at.strftime("%Y-%m-%d %H:%M UTC")

        message = f"""
🚨 **Nueva Señal**

{emoji} **{signal.symbol}** · {signal.timeframe}

{direction}

**Patrón:** {signal.pattern_name}
**Score:** {signal.score:.1f}
**Health:** {signal.health:.1f}%
**Probabilidad IA:** {signal.ml_probability:.0%}
**Fecha:** {timestamp}

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
        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }
        await self._post_with_retries(url, json=payload)

    async def _send_photo(
        self,
        message: str,
        candles: list[Candle],
        signal: Signal,
        pattern: Optional[PatternResult] = None,
    ) -> None:
        url = f"https://api.telegram.org/bot{self._bot_token}/sendPhoto"
        chart = self._chart_generator.create_candlestick_chart(
            candles,
            title=f"{signal.symbol} {signal.timeframe} — {signal.pattern_name}",
            patterns=[pattern] if pattern is not None else None,
        )
        png = chart.to_image(format="png")
        await self._post_with_retries(
            url,
            data={"chat_id": self._chat_id, "caption": message},
            files={"photo": ("chart.png", BytesIO(png), "image/png")},
        )

    async def _post_with_retries(self, url: str, **kwargs: Any) -> None:
        import httpx

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, **kwargs)
                    response.raise_for_status()
                    return
            except Exception as e:
                last_error = e
                if attempt < self._max_retries:
                    await asyncio.sleep(self._retry_backoff * (2**attempt))
        if last_error is not None:
            raise last_error
