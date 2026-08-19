from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np

from app.backtesting.models import MonteCarloResult, ValidationFold, ValidationResult
from app.core.logger import get_logger

logger = get_logger("BacktestValidation")

EvaluateFn = Callable[..., dict[str, Any]]


class TimeSeriesSplitter:
    """Genera divisiones temporales de un dataset indexado por posición."""

    @staticmethod
    def walk_forward(
        n: int,
        train_size: int,
        test_size: int,
        step: Optional[int] = None,
        min_train: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        step = step or test_size
        min_train = min_train or train_size
        folds: list[dict[str, Any]] = []
        start = 0
        index = 0
        while start + train_size + test_size <= n:
            train_start = start
            train_end = start + train_size
            test_start = train_end
            test_end = min(test_start + test_size, n)
            if test_end - test_start <= 0:
                break
            folds.append(
                {
                    "index": index,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "train_size": train_end - train_start,
                    "test_size": test_end - test_start,
                }
            )
            start += step
            index += 1
        return folds

    @staticmethod
    def rolling_window(
        n: int, window_size: int, step: Optional[int] = None
    ) -> list[dict[str, Any]]:
        step = step or window_size
        folds: list[dict[str, Any]] = []
        start = 0
        index = 0
        while start + window_size <= n:
            folds.append(
                {
                    "index": index,
                    "train_start": start,
                    "train_end": start,
                    "test_start": start,
                    "test_end": start + window_size,
                    "train_size": 0,
                    "test_size": window_size,
                }
            )
            start += step
            index += 1
        return folds

    @staticmethod
    def train_test_split(n: int, test_ratio: float = 0.3) -> dict[str, Any]:
        test_size = max(1, int(n * test_ratio))
        train_size = n - test_size
        return {
            "index": 0,
            "train_start": 0,
            "train_end": train_size,
            "test_start": train_size,
            "test_end": n,
            "train_size": train_size,
            "test_size": test_size,
        }

    @staticmethod
    def kfold(n: int, n_splits: int = 5) -> list[dict[str, Any]]:
        folds: list[dict[str, Any]] = []
        block = max(1, n // n_splits)
        for i in range(n_splits):
            test_start = i * block
            test_end = n if i == n_splits - 1 else (i + 1) * block
            train_indices = list(range(0, test_start)) + list(range(test_end, n))
            train_start = train_indices[0] if train_indices else test_end
            train_end = train_indices[-1] + 1 if train_indices else test_end
            folds.append(
                {
                    "index": i,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "train_size": len(train_indices),
                    "test_size": test_end - test_start,
                }
            )
        return folds


class BaseValidator:
    """Valida una estrategia sobre distintas particiones temporales."""

    method: str = "base"

    def __init__(self, evaluate_fn: Optional[EvaluateFn] = None) -> None:
        self._evaluate_fn = evaluate_fn

    def splits(self, n: int) -> list[dict[str, Any]]:
        raise NotImplementedError

    def run(
        self,
        candles: list[Any],
        evaluate_fn: Optional[EvaluateFn] = None,
    ) -> ValidationResult:
        evaluate = evaluate_fn or self._evaluate_fn
        if evaluate is None:
            raise ValueError("Se necesita un evaluate_fn que ejecute la estrategia")

        fold_specs = self.splits(len(candles))
        folds: list[ValidationFold] = []
        metric_keys: set[str] = set()

        for spec in fold_specs:
            test_candles = candles[spec["test_start"] : spec["test_end"]]
            metrics = evaluate(test_candles)
            metric_keys.update(metrics.keys())
            fold = ValidationFold(
                index=spec["index"],
                train_start=self._ts(candles, spec["train_start"], spec["train_end"]),
                train_end=self._ts(candles, spec["train_end"] - 1, spec["train_end"]),
                test_start=self._ts(candles, spec["test_start"], spec["test_end"]),
                test_end=self._ts(candles, spec["test_end"] - 1, spec["test_end"]),
                train_size=spec["train_size"],
                test_size=spec["test_size"],
                metrics=metrics,
            )
            folds.append(fold)

        aggregate = self._aggregate(folds, sorted(metric_keys))
        return ValidationResult(
            method=self.method,
            folds=folds,
            aggregate=aggregate,
            metadata={"total_candles": len(candles), "folds": len(folds)},
        )

    @staticmethod
    def _ts(candles: list[Any], index: int, bound: int) -> Optional[datetime]:
        if not candles or index < 0 or index >= bound or index >= len(candles):
            return None
        return candles[index].data.timestamp

    @staticmethod
    def _aggregate(folds: list[ValidationFold], keys: list[str]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if not folds:
            return result
        for key in keys:
            values = [float(f.metrics[key]) for f in folds if key in f.metrics]
            if not values:
                continue
            arr = np.array(values)
            result[key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()) if len(arr) > 1 else 0.0,
                "min": float(arr.min()),
                "max": float(arr.max()),
                "median": float(np.median(arr)),
            }
        return result


class WalkForwardValidator(BaseValidator):
    """Anchored walk-forward: entrenamiento acumulado + test consecutivo."""

    method = "walk_forward"

    def __init__(
        self,
        train_size: int,
        test_size: int,
        step: Optional[int] = None,
        min_train: Optional[int] = None,
        evaluate_fn: Optional[EvaluateFn] = None,
    ) -> None:
        super().__init__(evaluate_fn)
        self._train_size = train_size
        self._test_size = test_size
        self._step = step
        self._min_train = min_train

    def splits(self, n: int) -> list[dict[str, Any]]:
        return TimeSeriesSplitter.walk_forward(
            n, self._train_size, self._test_size, self._step, self._min_train
        )


class OutOfSampleValidator(BaseValidator):
    """División entrenamiento/test fuera de muestra."""

    method = "out_of_sample"

    def __init__(self, test_ratio: float = 0.3, evaluate_fn: Optional[EvaluateFn] = None) -> None:
        super().__init__(evaluate_fn)
        self._test_ratio = test_ratio

    def splits(self, n: int) -> list[dict[str, Any]]:
        return [TimeSeriesSplitter.train_test_split(n, self._test_ratio)]


class RollingWindowValidator(BaseValidator):
    """Ventanas rodantes de tamaño fijo."""

    method = "rolling_window"

    def __init__(
        self,
        window_size: int,
        step: Optional[int] = None,
        evaluate_fn: Optional[EvaluateFn] = None,
    ) -> None:
        super().__init__(evaluate_fn)
        self._window_size = window_size
        self._step = step

    def splits(self, n: int) -> list[dict[str, Any]]:
        return TimeSeriesSplitter.rolling_window(n, self._window_size, self._step)


class CrossValidator(BaseValidator):
    """Validación cruzada k-fold (bloques temporales)."""

    method = "cross_validation"

    def __init__(self, n_splits: int = 5, evaluate_fn: Optional[EvaluateFn] = None) -> None:
        super().__init__(evaluate_fn)
        self._n_splits = n_splits

    def splits(self, n: int) -> list[dict[str, Any]]:
        return TimeSeriesSplitter.kfold(n, self._n_splits)


class MonteCarloSimulator:
    """Re-muestreo de trades para estimar robustez de la estrategia."""

    def __init__(self, random_state: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(random_state)

    def simulate(
        self,
        trades: list[Any],
        n_simulations: int = 1000,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
    ) -> MonteCarloResult:
        pnls = np.array([t.pnl for t in trades], dtype=float)
        if pnls.size == 0:
            return MonteCarloResult(simulations=0, initial_capital=initial_capital)

        pnls = pnls - np.abs(pnls) * commission
        final_equities: list[float] = []
        max_drawdowns: list[float] = []

        for _ in range(n_simulations):
            sampled = self._rng.choice(pnls, size=len(pnls), replace=True)
            equity = initial_capital + np.cumsum(sampled)
            peak = np.maximum.accumulate(equity)
            dd = (peak - equity) / peak
            final_equities.append(float(equity[-1]))
            max_drawdowns.append(float(dd.max()) if len(dd) else 0.0)

        finals = np.array(final_equities)

        percentiles = {
            "p5": float(np.percentile(finals, 5)),
            "p25": float(np.percentile(finals, 25)),
            "p50": float(np.percentile(finals, 50)),
            "p75": float(np.percentile(finals, 75)),
            "p95": float(np.percentile(finals, 95)),
        }

        losses = finals[finals < initial_capital]
        cvar = float(losses.mean()) if losses.size else 0.0

        return MonteCarloResult(
            simulations=n_simulations,
            initial_capital=initial_capital,
            final_equities=final_equities,
            max_drawdowns=max_drawdowns,
            percentiles=percentiles,
            probability_of_profit=float(np.mean(finals > initial_capital)),
            var_95=float(np.percentile(finals, 5) - initial_capital),
            cvar_95=float(cvar - initial_capital),
            median_final_equity=float(np.median(finals)),
            mean_final_equity=float(finals.mean()),
            best_final_equity=float(finals.max()),
            worst_final_equity=float(finals.min()),
        )
