from app.strategy import strategies  # noqa: F401  (registra estrategias)
from app.strategy.base import BaseStrategy, StrategyDecision, StrategySignal
from app.strategy.engine import StrategyEngine, StrategyEngineResult
from app.strategy.evaluator import compare_strategies, run_strategy_backtest
from app.strategy.factory import StrategyFactory
from app.strategy.registry import StrategyRegistry, register_strategy

__all__ = [
    "BaseStrategy",
    "StrategyDecision",
    "StrategySignal",
    "StrategyEngine",
    "StrategyEngineResult",
    "StrategyFactory",
    "StrategyRegistry",
    "register_strategy",
    "run_strategy_backtest",
    "compare_strategies",
]
