from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, HTTPException

from app.backtesting.engine import BacktestEngine
from app.backtesting.models import (
    BacktestConfig,
    BacktestResult,
    Trade,
    TradeDirection,
    TradeStatus,
)
from app.backtesting.optimization import BacktestOptimizer
from app.backtesting.runner import BacktestRunner
from app.backtesting.validation import (
    CrossValidator,
    MonteCarloSimulator,
    OutOfSampleValidator,
    RollingWindowValidator,
    WalkForwardValidator,
)
from app.market.candles.models import Candle, CandleData
from app.patterns.base_pattern import PatternResult, PatternStatus, PatternType

router = APIRouter()

_backtests: list[BacktestResult] = []


def _generate_candles(n: int = 500, seed: int = 42) -> list[Candle]:
    rng = np.random.default_rng(seed)
    candles: list[Candle] = []
    base = 50000.0
    now = datetime.now(timezone.utc)
    for i in range(n):
        change = rng.normal(0.0005, 0.02) * base
        open_price = base
        close_price = open_price + change
        volatility = abs(change) * rng.uniform(0.5, 1.5)
        candles.append(
            Candle(
                symbol="BTCUSDT",
                timeframe="1h",
                data=CandleData(
                    timestamp=now - timedelta(hours=n - i),
                    open=open_price,
                    high=max(open_price, close_price) + rng.uniform(0, volatility),
                    low=min(open_price, close_price) - rng.uniform(0, volatility),
                    close=close_price,
                    volume=rng.integers(5000, 50000),
                ),
            )
        )
        base = close_price
    return candles


def _generate_patterns(candles: list[Candle], n: int = 30) -> list[PatternResult]:
    patterns: list[PatternResult] = []
    for i in range(n):
        idx = min(len(candles) - 1, i * 12)
        candle = candles[idx]
        patterns.append(
            PatternResult(
                pattern_name="double_top" if i % 2 else "double_bottom",
                pattern_type=PatternType.REVERSAL,
                symbol="BTCUSDT",
                timeframe="1h",
                direction=TradeDirection.SHORT if i % 2 else TradeDirection.LONG,
                confidence=0.8,
                status=PatternStatus.CONFIRMED,
                entry_price=candle.data.close,
                stop_loss=candle.data.close * 0.98,
                take_profit=candle.data.close * 1.06,
                detected_at=candle.data.timestamp,
                score=80.0,
            )
        )
    return patterns


def _resolve_backtest_input(payload: dict[str, Any]) -> tuple[list[Candle], list[PatternResult]]:
    candles = payload.get("candles")
    patterns = payload.get("patterns")
    if candles is None and patterns is None:
        candles = _generate_candles(payload.get("candles_count", 500))
        patterns = _generate_patterns(candles, payload.get("patterns_count", 30))
    else:
        raise HTTPException(
            status_code=400,
            detail="El payload debe incluir 'candles' y 'patterns', o ninguno (datos sintéticos)",
        )
    return candles, patterns


def _config_from_payload(payload: dict[str, Any]) -> BacktestConfig:
    return BacktestConfig(**{k: v for k, v in payload.get("config", {}).items()})


@router.get("/")
async def list_backtests():
    return {
        "backtests": [
            {
                "id": i,
                "start_date": bt.start_date.isoformat(),
                "end_date": bt.end_date.isoformat(),
                "total_trades": bt.metrics.total_trades,
                "win_rate": bt.metrics.win_rate,
                "profit_factor": bt.metrics.profit_factor,
                "sharpe_ratio": bt.metrics.sharpe_ratio,
                "total_pnl": bt.metrics.total_pnl,
                "total_return": bt.total_return,
            }
            for i, bt in enumerate(_backtests)
        ]
    }


@router.get("/{backtest_id}")
async def get_backtest(backtest_id: int):
    if backtest_id >= len(_backtests):
        raise HTTPException(status_code=404, detail="Backtest not found")

    bt = _backtests[backtest_id]
    return {
        "id": backtest_id,
        "config": bt.config.model_dump(),
        "metrics": bt.metrics.model_dump(),
        "trades_count": len(bt.trades),
        "equity_curve": bt.equity_curve[:100],
        "start_date": bt.start_date.isoformat(),
        "end_date": bt.end_date.isoformat(),
        "initial_capital": bt.initial_capital,
        "final_capital": bt.final_capital,
        "total_return": bt.total_return,
    }


@router.get("/{backtest_id}/trades")
async def get_backtest_trades(backtest_id: int):
    if backtest_id >= len(_backtests):
        raise HTTPException(status_code=404, detail="Backtest not found")

    bt = _backtests[backtest_id]
    return {
        "trades": [
            {
                "id": t.id,
                "symbol": t.symbol,
                "direction": t.direction.value,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "status": t.status.value,
            }
            for t in bt.trades
        ]
    }


@router.post("/runs")
async def run_backtest(payload: dict[str, Any]):
    candles, patterns = _resolve_backtest_input(payload)
    config = _config_from_payload(payload)
    runner = BacktestRunner(config)
    result = runner.run(candles, patterns)
    _backtests.append(result)
    return {
        "id": len(_backtests) - 1,
        "metrics": result.metrics.model_dump(),
        "total_return": result.total_return,
        "trades_count": len(result.trades),
    }


@router.post("/multi")
async def run_multiple(payload: dict[str, Any]):
    candles, patterns = _resolve_backtest_input(payload)
    configs = payload.get("configs") or [payload.get("config", {})]
    specs = []
    for i, cfg in enumerate(configs):
        specs.append(
            {
                "name": cfg.get("name", f"config_{i}"),
                "candles": candles,
                "patterns": patterns,
                "config": BacktestConfig(**{k: v for k, v in cfg.items() if k != "name"}),
            }
        )
    runner = BacktestRunner()
    results = runner.run_multiple(specs)
    for r in results:
        _backtests.append(r)
    return {"summary": runner.compare(results)}


def _validation_runner(payload: dict[str, Any]):
    candles, patterns = _resolve_backtest_input(payload)
    config = _config_from_payload(payload)

    def evaluate(test_candles: list[Candle]) -> dict[str, Any]:
        result = BacktestEngine(config).run(test_candles, patterns)
        return {
            "total_return": result.total_return,
            "win_rate": result.metrics.win_rate,
            "profit_factor": result.metrics.profit_factor,
            "sharpe_ratio": result.metrics.sharpe_ratio,
            "total_pnl": result.metrics.total_pnl,
        }

    return candles, evaluate


@router.post("/walk-forward")
async def walk_forward(payload: dict[str, Any]):
    candles, evaluate = _validation_runner(payload)
    validator = WalkForwardValidator(
        train_size=payload.get("train_size", 300),
        test_size=payload.get("test_size", 100),
        step=payload.get("step"),
        evaluate_fn=evaluate,
    )
    return validator.run(candles).model_dump(mode="json")


@router.post("/oos")
async def out_of_sample(payload: dict[str, Any]):
    candles, evaluate = _validation_runner(payload)
    validator = OutOfSampleValidator(
        test_ratio=payload.get("test_ratio", 0.3), evaluate_fn=evaluate
    )
    return validator.run(candles).model_dump(mode="json")


@router.post("/rolling")
async def rolling_window(payload: dict[str, Any]):
    candles, evaluate = _validation_runner(payload)
    validator = RollingWindowValidator(
        window_size=payload.get("window_size", 200),
        step=payload.get("step"),
        evaluate_fn=evaluate,
    )
    return validator.run(candles).model_dump(mode="json")


@router.post("/cross-validate")
async def cross_validate(payload: dict[str, Any]):
    candles, evaluate = _validation_runner(payload)
    validator = CrossValidator(
        n_splits=payload.get("n_splits", 5), evaluate_fn=evaluate
    )
    return validator.run(candles).model_dump(mode="json")


@router.post("/monte-carlo")
async def monte_carlo(payload: dict[str, Any]):
    candles, patterns = _resolve_backtest_input(payload)
    config = _config_from_payload(payload)
    result = BacktestEngine(config).run(candles, patterns)
    simulator = MonteCarloSimulator(random_state=payload.get("seed"))
    mc = simulator.simulate(
        trades=result.trades,
        n_simulations=payload.get("simulations", 1000),
        initial_capital=config.initial_capital,
    )
    return mc.model_dump(mode="json")


@router.post("/optimize")
async def optimize(payload: dict[str, Any]):
    candles, patterns = _resolve_backtest_input(payload)
    method = payload.get("method", "grid")
    config = _config_from_payload(payload)
    param_grid = payload.get("param_grid", {})
    metric = payload.get("metric", "sharpe_ratio")

    def objective(**params: Any) -> float:
        merged = {**config.model_dump(), **params}
        cfg = BacktestConfig(**{k: v for k, v in merged.items() if k in config.model_dump()})
        result = BacktestEngine(cfg).run(candles, patterns)
        return float(getattr(result.metrics, metric, 0.0))

    optimizer = BacktestOptimizer(random_state=payload.get("seed"))
    if method == "grid":
        return optimizer.grid_search(param_grid, objective)
    if method == "random":
        return optimizer.random_search(
            param_grid, objective, n_iter=payload.get("n_iter", 50)
        )
    if method == "bayesian":
        return optimizer.bayesian_optimization(
            param_grid, objective, n_trials=payload.get("n_trials", 50)
        )
    raise HTTPException(status_code=400, detail=f"Método desconocido: {method}")
