from __future__ import annotations

from typing import Any, Optional

from app.core.logger import get_logger
from app.patterns.hypothesis import PatternHypothesis
from app.strategy.base import StrategyDecision, StrategySignal
from app.strategy.factory import StrategyFactory
from app.strategy.registry import StrategyRegistry

logger = get_logger("StrategyEvaluator")


def run_strategy_backtest(
    hypotheses: list[PatternHypothesis],
    strategy_name: str,
    parameters: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Evalúa una estrategia sobre un histórico de hipótesis.

    Las hipótesis pueden traer un resultado posterior en
    ``market_structure["outcome"]`` (p.ej. ``{"pnl_pct": 3.5}``) para medir
    win rate. Si no lo traen, solo se reportan entradas y confianza media.
    """
    strategy = StrategyFactory.create(strategy_name, parameters)
    decisions: list[StrategyDecision] = []
    signals: list[StrategySignal] = []
    wins = 0
    outcomes = 0

    for hypothesis in hypotheses:
        decision = strategy.evaluate(hypothesis)
        decisions.append(decision)
        if decision.is_entry and decision.signal is not None:
            signals.append(decision.signal)
            outcome = hypothesis.market_structure.get("outcome")
            if isinstance(outcome, dict) and "pnl_pct" in outcome:
                outcomes += 1
                if outcome["pnl_pct"] > 0:
                    wins += 1

    win_rate = wins / outcomes if outcomes else None
    avg_confidence = (
        sum(s.confidence for s in signals) / len(signals) if signals else 0.0
    )

    result: dict[str, Any] = {
        "strategy": strategy_name,
        "hypotheses_evaluated": len(hypotheses),
        "entries": len(signals),
        "no_trades": len(decisions) - len(signals),
        "avg_confidence": avg_confidence,
        "directions": {
            "LONG": sum(1 for s in signals if s.direction == "LONG"),
            "SHORT": sum(1 for s in signals if s.direction == "SHORT"),
        },
        "win_rate": win_rate,
        "parameters": strategy.get_parameters(),
    }
    return result


def compare_strategies(
    hypotheses: list[PatternHypothesis],
    strategy_names: Optional[list[str]] = None,
    parameters: Optional[dict[str, dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Compara varias estrategias sobre las mismas hipótesis (backtest)."""
    parameters = parameters or {}
    names = strategy_names or list(StrategyRegistry.get_all().keys())
    return [
        run_strategy_backtest(hypotheses, name, parameters.get(name))
        for name in names
    ]
