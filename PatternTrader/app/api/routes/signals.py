from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_signal_repository
from app.database.repositories import SignalRepository
from app.signals.models import SignalPriority, SignalStatus

router = APIRouter()


@router.get("/")
async def list_signals(
    status: Optional[SignalStatus] = None,
    priority: Optional[SignalPriority] = None,
    symbol: Optional[str] = None,
    repo: SignalRepository = Depends(get_signal_repository),
):
    signals = await repo.list(status=status, priority=priority, symbol=symbol)
    return {
        "signals": [
            {
                "id": str(s.id),
                "symbol": s.symbol,
                "pattern": s.pattern_name,
                "direction": s.direction,
                "priority": s.priority.value,
                "status": s.status.value,
                "score": s.score,
                "entry_price": s.entry_price,
                "stop_loss": s.stop_loss,
                "take_profit": s.take_profit,
                "created_at": s.created_at.isoformat(),
            }
            for s in signals
        ]
    }


@router.get("/{signal_id}")
async def get_signal(
    signal_id: str,
    repo: SignalRepository = Depends(get_signal_repository),
):
    signal = await repo.get(signal_id)
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")

    return {
        "id": str(signal.id),
        "symbol": signal.symbol,
        "timeframe": signal.timeframe,
        "pattern": signal.pattern_name,
        "direction": signal.direction,
        "priority": signal.priority.value,
        "status": signal.status.value,
        "entry_price": signal.entry_price,
        "stop_loss": signal.stop_loss,
        "take_profit": signal.take_profit,
        "risk_reward_ratio": signal.risk_reward_ratio,
        "score": signal.score,
        "health": signal.health,
        "ml_probability": signal.ml_probability,
        "reasons": signal.reasons,
        "created_at": signal.created_at.isoformat(),
    }
