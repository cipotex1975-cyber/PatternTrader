from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from app.backtesting.engine import BacktestEngine
from app.backtesting.models import BacktestConfig, BacktestResult
from app.core.logger import get_logger

logger = get_logger("BacktestRunner")


class BacktestRunner:
    """Orquestador de backtests: simple, múltiple y paralelo."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self._config = config or BacktestConfig()

    def run(self, candles: list, patterns: list, config: BacktestConfig | None = None) -> BacktestResult:
        """Backtest simple sobre un conjunto de velas y patrones."""
        engine = BacktestEngine(config or self._config)
        result = engine.run(candles, patterns)
        result.metadata["name"] = "simple"
        return result

    def run_multiple(self, specs: list[dict[str, Any]]) -> list[BacktestResult]:
        """Backtests múltiples.

        Cada spec es un dict con las claves:
          - ``name`` (opcional): nombre identificativo
          - ``candles``: lista de velas
          - ``patterns``: lista de patrones
          - ``config`` (opcional): ``BacktestConfig`` específico
        """
        results: list[BacktestResult] = []
        for spec in specs:
            config = spec.get("config") or self._config
            engine = BacktestEngine(config)
            result = engine.run(spec["candles"], spec["patterns"])
            result.metadata["name"] = spec.get("name", f"backtest_{len(results)}")
            results.append(result)
        return results

    def run_parallel(
        self,
        specs: list[dict[str, Any]],
        max_workers: int = 4,
    ) -> list[BacktestResult]:
        """Backtests múltiples en paralelo (CPU-bound)."""
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(self._run_spec, spec) for spec in specs]
            results = [f.result() for f in futures]
        return results

    def _run_spec(self, spec: dict[str, Any]) -> BacktestResult:
        config = spec.get("config") or self._config
        engine = BacktestEngine(config)
        result = engine.run(spec["candles"], spec["patterns"])
        result.metadata["name"] = spec.get("name", "backtest")
        return result

    def compare(self, results: list[BacktestResult]) -> dict[str, list[dict[str, Any]]]:
        """Compara resultados y devuelve un ranking por métricas."""
        summary: list[dict[str, Any]] = []
        for result in results:
            name = result.metadata.get("name", "backtest")
            m = result.metrics
            summary.append(
                {
                    "name": name,
                    "total_return": result.total_return,
                    "total_pnl": m.total_pnl,
                    "win_rate": m.win_rate,
                    "profit_factor": m.profit_factor,
                    "sharpe_ratio": m.sharpe_ratio,
                    "sortino_ratio": m.sortino_ratio,
                    "max_drawdown_pct": m.max_drawdown_pct,
                    "expectancy": m.expectancy,
                    "trades": m.total_trades,
                }
            )

        def sort_key(metric: str) -> Callable[[dict[str, Any]], Any]:
            return lambda r: r[metric]

        return {
            "by_return": sorted(summary, key=sort_key("total_return"), reverse=True),
            "by_sharpe": sorted(summary, key=sort_key("sharpe_ratio"), reverse=True),
            "by_win_rate": sorted(summary, key=sort_key("win_rate"), reverse=True),
            "by_profit_factor": sorted(summary, key=sort_key("profit_factor"), reverse=True),
        }
