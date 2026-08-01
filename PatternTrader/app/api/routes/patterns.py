from fastapi import APIRouter, HTTPException

from app.patterns.registry import PatternRegistry

router = APIRouter()


@router.get("/")
async def list_patterns():
    patterns = PatternRegistry.get_all()
    return {
        "patterns": [
            {
                "name": name,
                "type": cls().pattern_type.value,
                "max_confirmation_candles": cls().max_confirmation_candles,
            }
            for name, cls in patterns.items()
        ]
    }


@router.get("/{pattern_name}")
async def get_pattern(pattern_name: str):
    pattern_class = PatternRegistry.get(pattern_name)
    if not pattern_class:
        raise HTTPException(status_code=404, detail=f"Pattern {pattern_name} not found")

    pattern = pattern_class()
    return {
        "name": pattern.name,
        "type": pattern.pattern_type.value,
        "max_confirmation_candles": pattern.max_confirmation_candles,
        "statistics": pattern.statistics(),
    }


@router.get("/{pattern_name}/statistics")
async def get_pattern_statistics(pattern_name: str):
    pattern_class = PatternRegistry.get(pattern_name)
    if not pattern_class:
        raise HTTPException(status_code=404, detail=f"Pattern {pattern_name} not found")

    pattern = pattern_class()
    return pattern.statistics()
