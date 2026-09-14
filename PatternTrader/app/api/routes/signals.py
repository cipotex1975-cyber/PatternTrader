from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_signal_repository
from app.database.repositories import SignalRepository
from app.signals.models import Signal, SignalPriority, SignalStatus

router = APIRouter()


@router.get("/")
async def list_signals(
    status: Optional[SignalStatus] = None,
    priority: Optional[SignalPriority] = None,
    symbol: Optional[str] = None,
    data_source: Optional[str] = None,
    include_expired: bool = False,
    repo: SignalRepository = Depends(get_signal_repository),
):
    if data_source is None:
        data_source = "live"
    signals = await repo.list(
        status=status, priority=priority, symbol=symbol, data_source=data_source
    )
    if not include_expired:
        signals = [s for s in signals if not _is_stale_pending(s)]
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
                "data_source": s.data_source,
                "created_at": s.created_at.isoformat(),
            }
            for s in signals
        ]
    }


def _is_stale_pending(signal: Signal) -> bool:
    """True si la señal sigue PENDING pero su TTL (expires_at) ya venció."""
    return signal.status == SignalStatus.PENDING and signal.is_expired


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
        "data_source": signal.data_source,
        "created_at": signal.created_at.isoformat(),
    }
