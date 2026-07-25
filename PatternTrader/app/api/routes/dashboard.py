from fastapi import APIRouter

from app.lifecycle.engine import LifecycleEngine
from app.lifecycle.models import LifecycleState

router = APIRouter()

_lifecycle_engine = LifecycleEngine()


@router.get("/overview")
async def get_dashboard_overview():
    stats = _lifecycle_engine.get_statistics()
    active = _lifecycle_engine.get_active()

    return {
        "statistics": stats,
        "active_patterns": len(active),
        "total_lifecycles": sum(stats.values()),
    }


@router.get("/active")
async def get_active_patterns():
    active = _lifecycle_engine.get_active()
    return {
        "patterns": [
            {
                "id": str(lc.id),
                "symbol": lc.symbol,
                "timeframe": lc.timeframe,
                "pattern": lc.pattern_name,
                "state": lc.current_state.value,
                "transitions": lc.total_transitions,
                "created_at": lc.created_at.isoformat(),
            }
            for lc in active
        ]
    }


@router.get("/by-state/{state}")
async def get_patterns_by_state(state: str):
    try:
        lifecycle_state = LifecycleState(state)
    except ValueError:
        return {"error": f"Invalid state: {state}"}

    patterns = _lifecycle_engine.get_by_state(lifecycle_state)
    return {
        "state": state,
        "count": len(patterns),
        "patterns": [
            {
                "id": str(lc.id),
                "symbol": lc.symbol,
                "timeframe": lc.timeframe,
                "pattern": lc.pattern_name,
                "created_at": lc.created_at.isoformat(),
            }
            for lc in patterns
        ],
    }


@router.get("/by-symbol/{symbol}")
async def get_patterns_by_symbol(symbol: str):
    patterns = _lifecycle_engine.get_by_symbol(symbol)
    return {
        "symbol": symbol,
        "count": len(patterns),
        "patterns": [
            {
                "id": str(lc.id),
                "timeframe": lc.timeframe,
                "pattern": lc.pattern_name,
                "state": lc.current_state.value,
                "created_at": lc.created_at.isoformat(),
            }
            for lc in patterns
        ],
    }
