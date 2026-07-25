from fastapi import APIRouter, HTTPException
from typing import Optional

from app.backtesting.models import BacktestResult, BacktestConfig

router = APIRouter()

_backtests: list[BacktestResult] = []


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
