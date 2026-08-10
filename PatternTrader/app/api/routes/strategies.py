from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_strategy_manager
from app.strategy.manager import StrategyManager

router = APIRouter()


@router.get("/")
async def list_strategies(manager: StrategyManager = Depends(get_strategy_manager)):
    return {"strategies": manager.list()}


@router.get("/{name}")
async def get_strategy(
    name: str,
    manager: StrategyManager = Depends(get_strategy_manager),
):
    for strategy in manager.list():
        if strategy["name"] == name:
            return {"strategy": strategy}
    raise HTTPException(status_code=404, detail="Strategy not found")


@router.patch("/{name}")
async def update_strategy(
    name: str,
    body: dict,
    manager: StrategyManager = Depends(get_strategy_manager),
):
    enabled: Optional[bool] = body.get("enabled")
    parameters: Optional[dict] = body.get("parameters") or body.get("params")

    if enabled is None and not parameters:
        raise HTTPException(status_code=422, detail="Provide 'enabled' and/or 'parameters'")

    if enabled is True:
        if not manager.enable(name):
            raise HTTPException(status_code=404, detail="Strategy not found")
    elif enabled is False:
        if not manager.disable(name):
            raise HTTPException(status_code=404, detail="Strategy not found")

    if parameters is not None:
        if not manager.set_params(name, parameters):
            raise HTTPException(status_code=404, detail="Strategy not found")

    for strategy in manager.list():
        if strategy["name"] == name:
            return {"strategy": strategy}
    raise HTTPException(status_code=404, detail="Strategy not found")
