from fastapi import APIRouter, HTTPException
from typing import Optional

from app.signals.models import Signal, SignalPriority, SignalStatus

router = APIRouter()

_signals: list[Signal] = []


@router.get("/")
async def list_signals(
    status: Optional[SignalStatus] = None,
    priority: Optional[SignalPriority] = None,
    symbol: Optional[str] = None,
):
    filtered = _signals

    if status:
        filtered = [s for s in filtered if s.status == status]
    if priority:
        filtered = [s for s in filtered if s.priority == priority]
    if symbol:
        filtered = [s for s in filtered if s.symbol == symbol]

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
            for s in filtered
        ]
    }


@router.get("/{signal_id}")
async def get_signal(signal_id: str):
    signal = next((s for s in _signals if str(s.id) == signal_id), None)
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
