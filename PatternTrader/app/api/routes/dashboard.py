from fastapi import APIRouter, Depends

from app.api.dependencies import get_pattern_service, get_trade_repository
from app.backtesting.models import TradeStatus
from app.database.repositories import TradeRepository
from app.lifecycle.models import LifecycleState
from app.patterns.service import PatternService

router = APIRouter()


@router.get("/overview")
async def get_dashboard_overview(
    service: PatternService = Depends(get_pattern_service),
    trades_repo: TradeRepository = Depends(get_trade_repository),
):
    lifecycle = service.pipeline.lifecycle
    stats = lifecycle.get_statistics()
    active = lifecycle.get_active()

    open_trades = await trades_repo.list(status=TradeStatus.OPEN)
    closed_trades = await trades_repo.list(status=TradeStatus.CLOSED)

    return {
        "statistics": stats,
        "active_patterns": len(active),
        "total_lifecycles": sum(stats.values()),
        "pipeline": service.pipeline.stats(),
        "open_trades": len(open_trades),
        "closed_trades": len(closed_trades),
    }


@router.get("/active")
async def get_active_patterns(
    service: PatternService = Depends(get_pattern_service),
):
    active = service.pipeline.lifecycle.get_active()
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
async def get_patterns_by_state(
    state: str,
    service: PatternService = Depends(get_pattern_service),
):
    try:
        lifecycle_state = LifecycleState(state)
    except ValueError:
        return {"error": f"Invalid state: {state}"}

    patterns = service.pipeline.lifecycle.get_by_state(lifecycle_state)
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
async def get_patterns_by_symbol(
    symbol: str,
    service: PatternService = Depends(get_pattern_service),
):
    patterns = service.pipeline.lifecycle.get_by_symbol(symbol)
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
