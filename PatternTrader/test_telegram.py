from __future__ import annotations

import argparse
import asyncio
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.core.config.settings import get_settings  # noqa: E402
from app.core.logger import get_logger  # noqa: E402
from app.market.candles.models import Candle, CandleData  # noqa: E402
from app.signals.models import Signal, SignalPriority  # noqa: E402
from app.telegram.notifier import TelegramNotifier  # noqa: E402

logger = get_logger("TestTelegramCLI")


def build_synthetic_candles(symbol: str, timeframe: str, count: int = 120) -> list[Candle]:
    candles: list[Candle] = []
    price = 1.3500
    start = datetime.now(timezone.utc) - timedelta(minutes=count)
    for i in range(count):
        drift = 0.0004 * math.sin(i / 8.0)
        open_price = price
        close_price = price + drift
        candles.append(
            Candle(
                symbol=symbol,
                timeframe=timeframe,
                data=CandleData(
                    timestamp=start + timedelta(minutes=i),
                    open=open_price,
                    high=max(open_price, close_price) + 0.0003,
                    low=min(open_price, close_price) - 0.0003,
                    close=close_price,
                    volume=1000.0 + i,
                ),
            )
        )
        price = close_price
    return candles


def build_synthetic_signal(symbol: str, timeframe: str) -> Signal:
    return Signal(
        symbol=symbol,
        timeframe=timeframe,
        pattern_name="TEST_PATTERN",
        direction="LONG",
        priority=SignalPriority.CRITICAL,
        entry_price=1.3550,
        stop_loss=1.3520,
        take_profit=1.3610,
        risk_reward_ratio=2.0,
        score=96.5,
        health=92.0,
        ml_probability=0.78,
        reasons=["Mensaje de prueba del sistema de notificaciones"],
    )


async def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Envía una notificación de prueba a Telegram para validar token, "
            "chat_id y conectividad."
        )
    )
    parser.add_argument("--symbol", default="USDCAD", help="Símbolo para la señal sintética")
    parser.add_argument("--timeframe", default="H1", help="Timeframe para la señal sintética")
    parser.add_argument(
        "--signal",
        action="store_true",
        help="Envía una señal sintética con gráfico en vez de un mensaje simple",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Fuerza el envío aunque TELEGRAM_ENABLED=false en .env (solo runtime)",
    )
    args = parser.parse_args(argv)

    telegram = get_settings().telegram
    print("Configuración de Telegram (leída del .env):")
    print(f"  bot_token cargado : {'sí' if telegram.bot_token else 'NO'}")
    print(f"  chat_id cargado   : {'sí' if telegram.chat_id else 'NO'}")
    print(f"  enabled           : {telegram.enabled}")
    print(f"  send_image        : {telegram.send_image}")
    print(f"  min_priority      : {telegram.min_priority}")

    if not telegram.bot_token or not telegram.chat_id:
        print("\nERROR: falta TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID en .env")
        sys.exit(1)

    if args.force:
        telegram.enabled = True
        print("\n--force: enabled=True aplicado solo en runtime (no modifica .env)")

    if not telegram.enabled:
        print(
            "\nTelegram está deshabilitado (TELEGRAM_ENABLED=false en .env). "
            "Actívalo en .env o usa --force para esta prueba."
        )
        sys.exit(1)

    notifier = TelegramNotifier()
    await notifier.initialize()

    if args.signal:
        signal = build_synthetic_signal(args.symbol, args.timeframe)
        candles = build_synthetic_candles(args.symbol, args.timeframe)
        ok = await notifier.send_signal(signal, candles=candles)
    else:
        ok = await notifier.send_message("✅ Prueba PatternTrader: notificaciones OK")

    if ok:
        print("\n✅ Enviado. Revisa tu chat de Telegram.")
    else:
        print("\n❌ No se pudo enviar. Revisa los logs de error arriba.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
