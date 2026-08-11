from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Sequence

import numpy as np

from app.backtesting.models import (
    BacktestMetrics,
    ClassificationMetrics,
    Trade,
)
from app.core.logger import get_logger

logger = get_logger("MetricsCalculator")

TRADING_DAYS = 252


def _safe_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values))


class MetricsCalculator:
    """Cálculo profesional de métricas de rendimiento y de clasificación."""

    @staticmethod
    def calculate(
        trades: list[Trade],
        equity_curve: list[dict],
        initial_capital: float,
        start_date: datetime,
        end_date: datetime,
    ) -> BacktestMetrics:
        closed = [t for t in trades if t.exit_time is not None]
        if not closed:
            return BacktestMetrics(total_trades=len(trades))

        wins = [t for t in closed if t.pnl > 0]
        losses = [t for t in closed if t.pnl < 0]
        breakeven = [t for t in closed if t.pnl == 0]

        total_pnl = sum(t.pnl for t in closed)
        win_rate = len(wins) / len(closed) if closed else 0.0

        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0.0
        avg_trade = total_pnl / len(closed) if closed else 0.0
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss else 0.0

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else (float("inf") if gross_profit > 0 else 0.0)
        )

        returns = np.array([t.pnl_pct for t in closed], dtype=float)
        mean_return = float(returns.mean()) if len(returns) else 0.0
        std_return = _safe_std(returns.tolist())
        sharpe = (mean_return / std_return) * np.sqrt(TRADING_DAYS) if std_return > 0 else 0.0

        downside = returns[returns < 0]
        downside_dev = _safe_std(downside.tolist()) if len(downside) else 0.0
        sortino = (mean_return / downside_dev) * np.sqrt(TRADING_DAYS) if downside_dev > 0 else 0.0

        expected_value = (
            (win_rate * avg_win) + ((1 - win_rate) * avg_loss) if closed else 0.0
        )

        risk_per_trade = [
            abs(t.entry_price - t.stop_loss) * t.size
            for t in closed
            if t.stop_loss is not None and t.entry_price
        ]
        expectancy_r = 0.0
        if risk_per_trade and len(risk_per_trade) == len(closed):
            r_values = [
                t.pnl / risk if risk > 0 else 0.0
                for t, risk in zip(closed, risk_per_trade)
            ]
            expectancy_r = float(np.mean(r_values)) if r_values else 0.0

        dd_stats = MetricsCalculator._drawdown_stats(equity_curve)

        equity_values = [e["equity"] for e in equity_curve]
        if equity_values:
            equity_arr = np.array(equity_values, dtype=float)
            annual_vol = _safe_std(np.diff(equity_arr) / equity_arr[:-1])
        else:
            annual_vol = 0.0

        days = (end_date - start_date).days
        final_capital = initial_capital + total_pnl
        if days > 0 and initial_capital > 0 and final_capital > 0:
            annual_return = (final_capital / initial_capital) ** (365 / days) - 1
        elif initial_capital > 0 and final_capital <= 0:
            annual_return = -1.0
        else:
            annual_return = 0.0

        max_dd_pct = dd_stats["max_drawdown_pct"]
        calmar = annual_return / (max_dd_pct / 100.0) if max_dd_pct > 0 else 0.0

        total_fees = MetricsCalculator._estimate_fees(closed)

        return BacktestMetrics(
            total_trades=len(closed),
            winning_trades=len(wins),
            losing_trades=len(losses),
            breakeven_trades=len(breakeven),
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            calmar_ratio=calmar,
            ulcer_index=dd_stats["ulcer_index"],
            max_drawdown=dd_stats["max_drawdown"],
            max_drawdown_pct=dd_stats["max_drawdown_pct"],
            average_drawdown_pct=dd_stats["average_drawdown_pct"],
            max_drawdown_duration=dd_stats["max_drawdown_duration"],
            average_win=avg_win,
            average_loss=avg_loss,
            average_trade=avg_trade,
            payoff_ratio=payoff_ratio,
            expectancy=expected_value,
            expectancy_r=expectancy_r,
            total_pnl=total_pnl,
            total_pnl_pct=(total_pnl / initial_capital * 100) if initial_capital else 0.0,
            annual_return=annual_return,
            annualized_volatility=annual_vol * np.sqrt(TRADING_DAYS),
            volatility=annual_vol,
            total_fees=total_fees,
        )

    @staticmethod
    def _drawdown_stats(equity_curve: list[dict]) -> dict[str, Any]:
        values = [e["equity"] for e in equity_curve]
        if not values:
            return {
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "average_drawdown_pct": 0.0,
                "max_drawdown_duration": 0,
                "ulcer_index": 0.0,
            }

        max_dd = 0.0
        max_dd_pct = 0.0
        running_max = values[0]
        dd_values = []
        current_peak = 0
        max_duration = 0

        for i, value in enumerate(values):
            if value > running_max:
                running_max = value
                current_peak = i
            dd_pct = (running_max - value) / running_max if running_max > 0 else 0.0
            dd_values.append(dd_pct)
            max_dd_pct = max(max_dd_pct, dd_pct)
            max_dd = max(max_dd, running_max - value)
            max_duration = max(max_duration, i - current_peak)

        return {
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd_pct * 100,
            "average_drawdown_pct": (float(np.mean(dd_values)) if dd_values else 0.0) * 100,
            "max_drawdown_duration": max_duration,
            "ulcer_index": (
                float(np.sqrt(np.mean(np.square(dd_values)))) * 100 if dd_values else 0.0
            ),
        }

    @staticmethod
    def _estimate_fees(closed: list[Trade]) -> float:
        return sum(getattr(t, "fees", 0.0) for t in closed)

    @staticmethod
    def classification_metrics(
        y_true: Sequence[int],
        y_pred: Sequence[int],
        y_proba: Optional[Sequence[float]] = None,
    ) -> ClassificationMetrics:
        """Métricas de clasificación (para evaluación de señales/modelos)."""
        y_true = list(y_true)
        y_pred = list(y_pred)
        n = len(y_true)
        if n == 0:
            return ClassificationMetrics()

        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

        accuracy = (tp + tn) / n if n else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        roc_auc = 0.0
        pr_auc = 0.0
        if y_proba is not None and len(y_proba) == n:
            probs = np.array(y_proba, dtype=float)
            labels = np.array(y_true, dtype=int)
            if len(np.unique(labels)) == 2:
                roc_auc = MetricsCalculator._auc(labels, probs)
                pr_auc = MetricsCalculator._average_precision(labels, probs)

        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=float(roc_auc),
            pr_auc=float(pr_auc),
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            confusion_matrix=[[tn, fp], [fn, tp]],
        )

    @staticmethod
    def _auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        order = np.argsort(y_proba)
        ranks = np.empty(len(y_true), dtype=float)
        ranks[order] = np.arange(1, len(y_true) + 1)
        pos = y_true == 1
        neg = y_true == 0
        n_pos = pos.sum()
        n_neg = neg.sum()
        if n_pos == 0 or n_neg == 0:
            return 0.0
        return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    @staticmethod
    def _average_precision(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        order = np.argsort(y_proba)[::-1]
        sorted_labels = y_true[order]
        cum_tp = np.cumsum(sorted_labels)
        precision = cum_tp / np.arange(1, len(sorted_labels) + 1)
        recall = cum_tp / sorted_labels.sum() if sorted_labels.sum() > 0 else 0
        ap = 0.0
        prev_recall = 0.0
        for p, r in zip(precision, recall):
            ap += (r - prev_recall) * p
            prev_recall = r
        return float(ap)
