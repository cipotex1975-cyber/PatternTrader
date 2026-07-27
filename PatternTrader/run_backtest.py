from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from app.backtesting.engine import BacktestEngine
from app.backtesting.models import BacktestConfig
from app.market.candles.models import Candle, CandleData
from app.patterns.base_pattern import PatternResult, PatternStatus
from app.patterns.registry import PatternRegistry
import app.patterns.reversal.double_top
import app.patterns.reversal.double_bottom
import app.patterns.reversal.head_and_shoulders
import app.patterns.reversal.inverse_head_and_shoulders
import app.patterns.continuation.bull_flag
import app.patterns.continuation.bear_flag
import app.patterns.continuation.bull_pennant
import app.patterns.continuation.bear_pennant
from datetime import datetime, timezone

SYMBOL = "USDCAD"
TIMEFRAME = "H1"


def load_candles(file_path: str, max_candles: int | None = None) -> list[Candle]:
    df = pd.read_csv(file_path, sep="\t")
    df.columns = [c.strip() for c in df.columns]

    if "DateTime" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["DateTime"] + " " + df["time"])
    elif "DateTime" in df.columns:
        df["datetime"] = pd.to_datetime(df["DateTime"])
    else:
        first_col = df.columns[0]
        df["datetime"] = pd.to_datetime(df[first_col])

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Tickvol": "tickvol", "Volume": "volume", "Spread": "spread",
    })

    for col in ["open", "high", "low", "close", "tickvol", "volume", "spread"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("datetime").reset_index(drop=True)

    if max_candles and len(df) > max_candles:
        df = df.tail(max_candles).reset_index(drop=True)

    candles = []
    for _, row in df.iterrows():
        candles.append(Candle(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            data=CandleData(
                timestamp=row["datetime"].to_pydatetime().replace(tzinfo=timezone.utc),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("tickvol", row.get("volume", 0))),
            ),
        ))

    return candles


def detect_all_patterns(
    candles: list[Candle],
    window: int = 200,
    step: int = 200,
    max_patterns: int = 200,
) -> list[PatternResult]:
    import time

    detectors = PatternRegistry.get_all_instances()
    seen_keys: set[tuple[str, int]] = set()
    all_patterns: list[PatternResult] = []

    total_windows = (len(candles) - window) // step + 1
    print(f"  Ventanas a analizar: {total_windows}")

    start_time = time.time()
    last_print = start_time

    for win_idx, start in enumerate(range(0, len(candles) - window + 1, step)):
        if len(all_patterns) >= max_patterns:
            print(f"  Limite de {max_patterns} patrones alcanzado en ventana {win_idx}")
            break

        now = time.time()
        if win_idx % 50 == 0 or (now - last_print) >= 5:
            elapsed = now - start_time
            if win_idx > 0:
                avg = elapsed / win_idx
                remaining = avg * (total_windows - win_idx)
                m, s = divmod(int(remaining), 60)
                h, m = divmod(m, 60)
                eta = f"{h:02d}:{m:02d}:{s:02d}"
                print(f"  Ventana {win_idx}/{total_windows} | "
                      f"Patrones: {len(all_patterns)} | "
                      f"ETA: {eta}")
            else:
                print(f"  Ventana 0/{total_windows}...")
            last_print = now

        end = start + window
        window_candles = candles[start:end]

        for detector in detectors:
            result = detector.detect(window_candles, SYMBOL, TIMEFRAME)
            if result is None:
                continue

            dedup_key = (result.pattern_name, start // step)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            result.detected_at = window_candles[-1].data.timestamp
            result.entry_price = window_candles[-1].data.close
            all_patterns.append(result)

    return all_patterns


def prepare_patterns_for_backtest(patterns: list[PatternResult]) -> list[PatternResult]:
    prepared: list[PatternResult] = []

    for pattern in patterns:
        if not pattern.key_levels:
            continue

        sl, tp = _derive_sl_tp(pattern)
        if sl is None or tp is None:
            continue

        pattern.stop_loss = sl
        pattern.take_profit = tp
        pattern.status = PatternStatus.CONFIRMED
        prepared.append(pattern)

    return prepared


def _derive_sl_tp(pattern: PatternResult) -> tuple[float | None, float | None]:
    kl = pattern.key_levels
    name = pattern.pattern_name

    if name == "double_top":
        peak1 = kl.get("peak1", 0)
        peak2 = kl.get("peak2", 0)
        target = kl.get("target")
        if peak1 and peak2 and target:
            return max(peak1, peak2) * 1.002, target

    if name == "double_bottom":
        neckline = kl.get("neckline")
        target = kl.get("target")
        trough = min(kl.get("trough1", float("inf")), kl.get("trough2", float("inf")))
        if neckline and target:
            return trough * 0.998, target

    if name in ("bull_flag", "bull_pennant"):
        flag_low = kl.get("flag_low")
        target = kl.get("target")
        if flag_low and target:
            return flag_low * 0.998, target

    if name in ("bear_flag", "bear_pennant"):
        flag_high = kl.get("flag_high")
        target = kl.get("target")
        if flag_high and target:
            return flag_high * 1.002, target

    if name == "head_and_shoulders":
        head = kl.get("head", 0)
        target = kl.get("target")
        if head and target:
            return head * 1.002, target

    if name == "inverse_head_and_shoulders":
        neckline = kl.get("neckline")
        target = kl.get("target")
        if neckline and target:
            return neckline * 0.998, target

    return None, None


def print_results(result) -> None:
    print("=" * 70)
    print("RESULTADOS DEL BACKTEST")
    print("=" * 70)
    print(f"Par:              {SYMBOL} ({TIMEFRAME})")
    print(f"Periodo:          {result.start_date.date()} a {result.end_date.date()}")
    print(f"Capital inicial:  ${result.initial_capital:,.2f}")
    print(f"Capital final:    ${result.final_capital:,.2f}")
    print(f"Retorno total:    {result.total_return:.2%}")
    print()
    print("METRICAS DE TRADING")
    print("-" * 40)
    m = result.metrics
    print(f"  Trades totales:    {m.total_trades}")
    print(f"  Trades ganadores:  {m.winning_trades}")
    print(f"  Trades perdedores: {m.losing_trades}")
    print(f"  Win Rate:          {m.win_rate:.2%}")
    print(f"  Profit Factor:     {m.profit_factor:.2f}")
    print()
    print("METRICAS DE RIESGO")
    print("-" * 40)
    print(f"  Sharpe Ratio:      {m.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:      ${m.max_drawdown:,.2f} ({m.max_drawdown_pct:.2f}%)")
    print(f"  Expectancy:        ${m.expectancy:,.2f}")
    print(f"  Avg Win:           ${m.average_win:,.2f}")
    print(f"  Avg Loss:          ${m.average_loss:,.2f}")
    print()

    if result.trades:
        print("ANALISIS POR PATRON")
        print("-" * 60)
        by_pattern = defaultdict(lambda: {"count": 0, "wins": 0, "pnl": 0.0})
        for t in result.trades:
            p = t.pattern_name
            by_pattern[p]["count"] += 1
            by_pattern[p]["pnl"] += t.pnl
            if t.pnl > 0:
                by_pattern[p]["wins"] += 1

        for pat, data in sorted(by_pattern.items(), key=lambda x: -x[1]["count"]):
            wr = data["wins"] / data["count"] if data["count"] else 0
            print(f"  {pat:30s}  trades={data['count']:3d}  "
                  f"WR={wr:6.1%}  PnL=${data['pnl']:>10,.2f}")
        print()

        print("MEJORES 5 TRADES")
        print("-" * 60)
        for t in sorted(result.trades, key=lambda x: -x.pnl)[:5]:
            print(f"  {t.pattern_name:25s}  entry={t.entry_price:.5f}  "
                  f"exit={t.exit_price:.5f}  PnL=${t.pnl:>10,.2f}")

        print()
        print("PEORES 5 TRADES")
        print("-" * 60)
        for t in sorted(result.trades, key=lambda x: x.pnl)[:5]:
            print(f"  {t.pattern_name:25s}  entry={t.entry_price:.5f}  "
                  f"exit={t.exit_price:.5f}  PnL=${t.pnl:>10,.2f}")


def main() -> None:
    import argparse

    default_data = Path(__file__).parent / "app" / "datos_test" / "USDCAD_H1_201005311000_202606010000.txt"

    parser = argparse.ArgumentParser(description="Backtest de patrones chartistas")
    parser.add_argument(
        "--max-candles", type=int, default=None,
        help="Numero maximo de velas a usar (default: todas)",
    )
    parser.add_argument(
        "--data", type=str, default=str(default_data),
        help=f"Ruta al archivo de datos (default: {default_data.name})",
    )
    parser.add_argument(
        "--step", type=int, default=100,
        help="Paso entre ventanas deslizantes (default: 100, mayor = mas rapido)",
    )
    parser.add_argument(
        "--max-patterns", type=int, default=500,
        help="Numero maximo de patrones a detectar (default: 500)",
    )
    parser.add_argument(
        "--min-rr", type=float, default=1.0,
        help="Ratio Riesgo/Recompensa minimo para aceptar un trade (default: 1.0)",
    )
    parser.add_argument(
        "--exclude", type=str, default=None,
        help="Patrones a excluir, separados por coma (ej: bear_flag,double_top)",
    )
    args = parser.parse_args()

    data_path = Path(args.data)

    max_candles = args.max_candles

    print(f"Cargando datos desde: {data_path}")
    if max_candles:
        print(f"Usando ultimas {max_candles} velas")
    else:
        print("Usando todos los datos")
    candles = load_candles(str(data_path), max_candles=max_candles)
    print(f"Velas cargadas: {len(candles)}")
    print(f"Rango: {candles[0].data.timestamp.date()} a {candles[-1].data.timestamp.date()}")

    print(f"\nDetectando patrones (ventana deslizante, step={args.step})...")
    raw_patterns = detect_all_patterns(
        candles, window=200, step=args.step, max_patterns=args.max_patterns,
    )

    if args.exclude:
        excluded = set(args.exclude.split(","))
        raw_patterns = [p for p in raw_patterns if p.pattern_name not in excluded]
        print(f"Patrones excluidos: {', '.join(sorted(excluded))}")

    print(f"Patrones detectados: {len(raw_patterns)}")

    if raw_patterns:
        counts = defaultdict(int)
        for p in raw_patterns:
            counts[p.pattern_name] += 1
        for name, cnt in sorted(counts.items()):
            print(f"  {name}: {cnt}")

    patterns = prepare_patterns_for_backtest(raw_patterns)
    print(f"\nPatrones con SL/TP validos: {len(patterns)}")

    if not patterns:
        print("\nNo se encontraron patrones validos para backtest.")
        if raw_patterns:
            print("Detecciones raw sin preparar:")
            for p in raw_patterns[:5]:
                print(f"  {p.pattern_name}: key_levels={p.key_levels}, confidence={p.confidence:.2f}")
        return

    config = BacktestConfig(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005,
        max_positions=10,
        risk_per_trade=0.02,
    )

    print("\nEjecutando backtest...")
    engine = BacktestEngine(config)
    result = engine.run(candles, patterns)

    print()
    print_results(result)


if __name__ == "__main__":
    main()
