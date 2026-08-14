from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import defaultdict
from datetime import timezone
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from app.core.config.settings import get_settings
from app.core.events.bus import get_event_bus
from app.core.events.models import Event, EventType
from app.core.logger import get_logger
from app.learning.service import LearningService
from app.market.candles.models import Candle, CandleData
from app.patterns.pipeline import PatternPipeline
from train_and_compare import derive_symbol, derive_timeframe

logger = get_logger("SimulatePipeline")


def load_candles(file_path: str, symbol: str, timeframe: str) -> list[Candle]:
    """Carga un archivo OHLCV (MT4 tab-delimited o CSV con coma) como Candle[].

    Detecta el separador desde la primera línea y ordena cronológicamente,
    igual que ``run_backtest.load_candles`` pero sin el límite de ``max_candles``.
    """
    with open(file_path, "r") as f:
        first_line = f.readline()
    sep = "\t" if "\t" in first_line else ","

    df = pd.read_csv(file_path, sep=sep)
    df.columns = [c.strip() for c in df.columns]

    if "DateTime" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["DateTime"] + " " + df["time"])
    elif "DateTime" in df.columns:
        df["datetime"] = pd.to_datetime(df["DateTime"])
    else:
        first_col = df.columns[0]
        df["datetime"] = pd.to_datetime(df[first_col])

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Tickvol": "tickvol",
            "Volume": "volume",
            "Spread": "spread",
        }
    )

    for col in ["open", "high", "low", "close", "tickvol", "volume", "spread"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("datetime").reset_index(drop=True)

    candles: list[Candle] = []
    for _, row in df.iterrows():
        candles.append(
            Candle(
                symbol=symbol,
                timeframe=timeframe,
                data=CandleData(
                    timestamp=row["datetime"].to_pydatetime().replace(tzinfo=timezone.utc),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("tickvol", row.get("volume", 0))),
                ),
            )
        )
    return candles


async def _build_repositories(use_db: bool) -> dict[str, Any]:
    if not use_db:
        logger.info("Modo memoria: sin persistencia en base de datos")
        return {"lifecycle": None, "signal": None, "trade": None}

    from app.database.base import init_db
    from app.database.repositories import (
        LifecycleRepository,
        SignalRepository,
        TradeRepository,
    )

    try:
        await init_db()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"No se pudo conectar a PostgreSQL: {e}. "
            "Levanta la base con `docker compose up -d` o usa `--memory`."
        ) from e
    return {
        "lifecycle": LifecycleRepository(),
        "signal": SignalRepository(),
        "trade": TradeRepository(),
    }


class _EventRecorder:
    """Suscripciones al bus para capturar el flujo para el reporte."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = defaultdict(int)
        self.signals: list[dict[str, Any]] = []
        self.patterns: dict[str, int] = defaultdict(int)
        self._bus = get_event_bus()

    async def _on_detected(self, event: Event) -> None:
        self.counts["PATTERN_DETECTED"] += 1
        name = event.data.get("pattern_name", "unknown")
        self.patterns[name] += 1

    async def _on_confirmed(self, event: Event) -> None:
        self.counts["PATTERN_CONFIRMED"] += 1

    async def _on_signal_sent(self, event: Event) -> None:
        self.counts["SIGNAL_SENT"] += 1
        self.signals.append(dict(event.data))

    async def start(self) -> None:
        await self._bus.start()
        self._bus.subscribe(EventType.PATTERN_DETECTED, self._on_detected)
        self._bus.subscribe(EventType.PATTERN_CONFIRMED, self._on_confirmed)
        self._bus.subscribe(EventType.SIGNAL_SENT, self._on_signal_sent)

    async def stop(self) -> None:
        self._bus.unsubscribe(EventType.PATTERN_DETECTED, self._on_detected)
        self._bus.unsubscribe(EventType.PATTERN_CONFIRMED, self._on_confirmed)
        self._bus.unsubscribe(EventType.SIGNAL_SENT, self._on_signal_sent)
        await self._bus.stop()


async def run_simulation(
    data_file: str,
    *,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    warmup: int = 200,
    step: int = 50,
    max_candles: int = 500,
    speed: float = 0.0,
    use_db: bool = True,
    learning: bool = False,
    strategies: Optional[list[str]] = None,
    model_dir: Optional[str] = None,
    force_telegram: bool = False,
) -> dict[str, Any]:
    """Reproduce el flujo completo del pipeline sobre un archivo OHLCV.

    Devuelve el reporte de la simulación (stats, eventos y señales) para poder
    mostrarlo en consola o reutilizarlo desde tests.
    """
    symbol = symbol or derive_symbol(data_file)
    timeframe = timeframe or derive_timeframe(data_file)

    logger.info(f"Cargando velas desde {data_file}")
    candles = load_candles(data_file, symbol, timeframe)
    if len(candles) < warmup:
        raise ValueError(
            f"Archivo con solo {len(candles)} velas; se necesitan al menos {warmup} "
            f"(--warmup) para iniciar la simulación"
        )

    settings = get_settings()
    original_telegram = settings.telegram.enabled
    original_model_path = settings.ml.model_path
    model_dir_used = model_dir or original_model_path
    if force_telegram:
        settings.telegram.enabled = True
    if model_dir is not None:
        settings.ml.model_path = model_dir

    reporter = _EventRecorder()
    learning_service: Optional[LearningService] = None
    pipeline: Optional[PatternPipeline] = None

    try:
        await reporter.start()

        repos = await _build_repositories(use_db)

        if learning:
            from app.database.repositories import MLModelRepository
            from app.learning.repository import KnowledgeRepository

            learning_service = LearningService(
                repository=KnowledgeRepository(),
                ml_model_repository=MLModelRepository(),
            )
            await learning_service.start()

        pipeline = PatternPipeline(
            max_candles=max_candles,
            strategies=strategies,
            learning_service=learning_service,
            lifecycle_repository=repos["lifecycle"],
            signal_repository=repos["signal"],
        )

        total = len(candles)
        window = candles[:warmup]
        start_time = time.monotonic()
        last_log = start_time

        ml_models = pipeline.scoring.ensure_models([symbol])

        stats = await pipeline.process_symbol(symbol, timeframe, candles=window)
        ticks = 1
        end = warmup
        for end in range(warmup + step, total + 1, step):
            window = candles[:end]
            stats = await pipeline.process_symbol(symbol, timeframe, candles=window)
            ticks += 1

            now = time.monotonic()
            if now - last_log >= 5.0:
                elapsed = now - start_time
                remaining = elapsed / end * (total - end)
                m, s = divmod(int(remaining), 60)
                logger.info(
                    f"Tick {ticks} | velas {end}/{total} | "
                    f"tracked={stats.get('tracked', 0)} | "
                    f"señales={stats.get('signals_sent', 0)} | ETA {m:02d}:{s:02d}"
                )
                last_log = now

            if speed > 0:
                await asyncio.sleep(speed)

        if end < total:
            stats = await pipeline.process_symbol(symbol, timeframe, candles=candles)
            ticks += 1

        await asyncio.sleep(0.3)
        signals_report = list(reporter.signals)
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "data_file": data_file,
            "total_candles": total,
            "warmup": warmup,
            "step": step,
            "ticks": ticks,
            "elapsed_seconds": round(time.monotonic() - start_time, 2),
            "use_db": use_db,
            "learning": learning,
            "strategies": strategies or get_settings().strategies.enabled,
            "model_dir": model_dir_used,
            "telegram_enabled": get_settings().telegram.enabled,
            "ml_models": ml_models,
            "stats": stats,
            "events": dict(reporter.counts),
            "patterns_by_type": dict(reporter.patterns),
            "signals": signals_report,
        }
    finally:
        await reporter.stop()
        if learning_service is not None:
            await learning_service.stop()
        settings.telegram.enabled = original_telegram
        settings.ml.model_path = original_model_path


def _format_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    sep = "=" * 72
    lines.append(sep)
    lines.append(f"SIMULACIÓN DEL PIPELINE: {report['symbol']} {report['timeframe']}")
    lines.append(sep)
    lines.append(f"Archivo:     {report['data_file']}")
    lines.append(f"Velas:       {report['total_candles']}")
    lines.append(f"Warmup:      {report['warmup']} | Step: {report['step']} | Ticks: {report['ticks']}")
    lines.append(f"Duración:    {report['elapsed_seconds']}s")
    lines.append(f"Persistencia: {'PostgreSQL (DB)' if report['use_db'] else 'Memoria'}")
    lines.append(f"Aprendizaje: {'SÍ' if report['learning'] else 'no'}")
    lines.append(f"Estrategias: {', '.join(report['strategies'])}")
    lines.append(f"Modelo dir:  {report['model_dir']}")
    lines.append(f"Telegram:    {'activado' if report['telegram_enabled'] else 'desactivado (no-op)'}")

    if report["ml_models"]:
        lines.append("Modelos ML por símbolo:")
        for sym, name in sorted(report["ml_models"].items()):
            lines.append(f"  {sym}: {name}")
    else:
        lines.append(
            "Modelos ML por símbolo: ninguno cargado (fallback neutro). "
            "Entrena con `train_and_compare.py` para el par."
        )

    lines.append("")
    lines.append("EVENTOS CAPTURADOS")
    lines.append("-" * 40)
    for name in ("PATTERN_DETECTED", "PATTERN_CONFIRMED", "SIGNAL_SENT"):
        lines.append(f"  {name:20s}: {report['events'].get(name, 0)}")

    lines.append("")
    lines.append("PATRONES DETECTADOS")
    lines.append("-" * 40)
    if report["patterns_by_type"]:
        for name, count in sorted(report["patterns_by_type"].items(), key=lambda x: -x[1]):
            lines.append(f"  {name:30s}: {count}")
    else:
        lines.append("  (ninguno)")

    lines.append("")
    lines.append("ESTADÍSTICAS DEL PIPELINE")
    lines.append("-" * 40)
    for key, value in report["stats"].items():
        lines.append(f"  {key:20s}: {value}")

    signals = report["signals"]
    lines.append("")
    lines.append(f"SEÑALES ENVIADAS ({len(signals)})")
    lines.append("-" * 40)
    if not signals:
        lines.append("  (ninguna)")
    for i, signal in enumerate(signals, start=1):
        lines.append(
            f"  [{i}] {signal.get('symbol')} {signal.get('timeframe')} "
            f"{signal.get('direction')} {signal.get('pattern_name')} "
            f"score={signal.get('score', 0):.1f} "
            f"entry={signal.get('entry_price', 0):.5f} "
            f"sl={signal.get('stop_loss', 0):.5f} "
            f"tp={signal.get('take_profit', 0):.5f} "
            f"rr={signal.get('risk_reward_ratio', 0):.2f} "
            f"strategy={signal.get('strategy', '')}"
        )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simula el flujo completo del pipeline (detección → señal → Telegram) "
        "sobre un archivo OHLCV histórico."
    )
    parser.add_argument("data_file", type=str, help="Archivo OHLCV (tab o coma), ej: app/datos_test/USDCAD_H1_*.txt")
    parser.add_argument("--symbol", type=str, default=None, help="Símbolo (default: derivado del nombre)")
    parser.add_argument("--timeframe", type=str, default=None, help="Timeframe (default: derivado del nombre)")
    parser.add_argument("--warmup", type=int, default=200, help="Velas iniciales antes del replay (default: 200)")
    parser.add_argument("--step", type=int, default=50, help="Velas nuevas por tick del replay (default: 50)")
    parser.add_argument("--max-candles", type=int, default=500, help="Ventana máxima por tick (default: 500)")
    parser.add_argument("--speed", type=float, default=0.0, help="Pausa en segundos entre ticks (default: 0)")
    parser.add_argument("--memory", action="store_true", help="Ejecuta en memoria sin persistir en PostgreSQL")
    parser.add_argument("--learning", action="store_true", help="Activa el aprendizaje continuo (LearningService)")
    parser.add_argument("--strategy", type=str, default=None, help="Estrategias separadas por coma (default: settings)")
    parser.add_argument("--model-dir", type=str, default=None, help="Directorio de modelos ML (default: settings.ml.model_path)")
    parser.add_argument("--telegram", action="store_true", help="Fuerza el envío de notificaciones por Telegram")
    parser.add_argument("--quiet", action="store_true", help="Suprime logs de componentes (solo reporte final)")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.quiet:
        from app.core.logger import setup_logger

        get_settings().logging.level = "ERROR"
        setup_logger()
    strategies = args.strategy.split(",") if args.strategy else None
    report = asyncio.run(
        run_simulation(
            args.data_file,
            symbol=args.symbol,
            timeframe=args.timeframe,
            warmup=args.warmup,
            step=args.step,
            max_candles=args.max_candles,
            speed=args.speed,
            use_db=not args.memory,
            learning=args.learning,
            strategies=strategies,
            model_dir=args.model_dir,
            force_telegram=args.telegram,
        )
    )
    print(_format_report(report))


if __name__ == "__main__":
    main()
