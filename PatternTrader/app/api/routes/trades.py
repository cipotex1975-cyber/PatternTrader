from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_trade_repository
from app.backtesting.models import TradeStatus
from app.database.repositories import TradeRepository

router = APIRouter()


@router.get("/")
async def list_trades(
    status: Optional[TradeStatus] = None,
    symbol: Optional[str] = None,
    repo: TradeRepository = Depends(get_trade_repository),
):
    trades = await repo.list(status=status, symbol=symbol)
    return {
        "trades": [
            {
                "id": t.id,
                "symbol": t.symbol,
                "timeframe": t.timeframe,
                "direction": t.direction.value,
                "entry_price": t.entry_price,
                "entry_time": t.entry_time.isoformat(),
                "exit_price": t.exit_price,
                "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "size": t.size,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "status": t.status.value,
                "pattern_name": t.pattern_name,
            }
            for t in trades
        ]
    }


@router.get("/{trade_id}")
async def get_trade(
    trade_id: str,
    repo: TradeRepository = Depends(get_trade_repository),
):
    trade = await repo.get(trade_id)
    if trade is None:
        raise HTTPException(status_code=404, detail="Trade not found")

    return {
        "id": trade.id,
        "symbol": trade.symbol,
        "timeframe": trade.timeframe,
        "direction": trade.direction.value,
        "entry_price": trade.entry_price,
        "entry_time": trade.entry_time.isoformat(),
        "exit_price": trade.exit_price,
        "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
        "stop_loss": trade.stop_loss,
        "take_profit": trade.take_profit,
        "size": trade.size,
        "pnl": trade.pnl,
        "pnl_pct": trade.pnl_pct,
        "status": trade.status.value,
        "pattern_name": trade.pattern_name,
        "score": trade.score,
        "duration_seconds": trade.duration,
        "metadata": trade.metadata,
    }
