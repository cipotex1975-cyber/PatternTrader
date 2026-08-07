from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_pattern_service
from app.lifecycle.models import LifecycleState
from app.patterns.service import PatternService

router = APIRouter()


@router.get("/statistics")
async def get_statistics(
    service: PatternService = Depends(get_pattern_service),
):
    lifecycle = service.pipeline.lifecycle
    stats = lifecycle.get_statistics()
    return {
        "statistics": stats,
        "active": len(lifecycle.get_active()),
        "total": sum(stats.values()),
    }


@router.get("/")
async def list_lifecycles(
    state: Optional[LifecycleState] = None,
    symbol: Optional[str] = None,
    active: Optional[bool] = None,
    service: PatternService = Depends(get_pattern_service),
):
    lifecycle = service.pipeline.lifecycle
    result = lifecycle.get_active() if active else lifecycle.get_all()
    if state is not None:
        result = [lc for lc in result if lc.current_state == state]
    if symbol is not None:
        result = [lc for lc in result if lc.symbol == symbol]

    return {
        "lifecycles": [
            {
                "id": str(lc.id),
                "pattern_id": str(lc.pattern_id),
                "symbol": lc.symbol,
                "timeframe": lc.timeframe,
                "pattern": lc.pattern_name,
                "state": lc.current_state.value,
                "transitions": lc.total_transitions,
                "is_active": lc.is_active,
                "created_at": lc.created_at.isoformat(),
            }
            for lc in result
        ]
    }


@router.get("/pattern/{pattern_id}")
async def get_lifecycle_by_pattern(
    pattern_id: UUID,
    service: PatternService = Depends(get_pattern_service),
):
    lc = service.pipeline.lifecycle.get_by_pattern(pattern_id)
    if lc is None:
        raise HTTPException(status_code=404, detail="Lifecycle not found for pattern")

    return {
        "id": str(lc.id),
        "pattern_id": str(lc.pattern_id),
        "symbol": lc.symbol,
        "timeframe": lc.timeframe,
        "pattern": lc.pattern_name,
        "state": lc.current_state.value,
        "is_active": lc.is_active,
        "created_at": lc.created_at.isoformat(),
        "updated_at": lc.updated_at.isoformat(),
        "closed_at": lc.closed_at.isoformat() if lc.closed_at else None,
        "transitions": [
            {
                "from_state": t.from_state.value,
                "to_state": t.to_state.value,
                "timestamp": t.timestamp.isoformat(),
                "reason": t.reason,
                "metadata": t.metadata,
            }
            for t in lc.transitions
        ],
    }


@router.get("/{lifecycle_id}")
async def get_lifecycle(
    lifecycle_id: UUID,
    service: PatternService = Depends(get_pattern_service),
):
    lc = service.pipeline.lifecycle.get(lifecycle_id)
    if lc is None:
        raise HTTPException(status_code=404, detail="Lifecycle not found")

    return {
        "id": str(lc.id),
        "pattern_id": str(lc.pattern_id),
        "symbol": lc.symbol,
        "timeframe": lc.timeframe,
        "pattern": lc.pattern_name,
        "state": lc.current_state.value,
        "is_active": lc.is_active,
        "created_at": lc.created_at.isoformat(),
        "updated_at": lc.updated_at.isoformat(),
        "closed_at": lc.closed_at.isoformat() if lc.closed_at else None,
        "transitions": [
            {
                "from_state": t.from_state.value,
                "to_state": t.to_state.value,
                "timestamp": t.timestamp.isoformat(),
                "reason": t.reason,
                "metadata": t.metadata,
            }
            for t in lc.transitions
        ],
    }
