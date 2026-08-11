from __future__ import annotations

import itertools
import random
from typing import Any, Callable, Optional

from app.core.logger import get_logger

logger = get_logger("BacktestOptimizer")

ObjectiveFn = Callable[..., float]


class BacktestOptimizer:
    """Optimización de parámetros de estrategia: grid, random y bayesiana."""

    def __init__(self, random_state: Optional[int] = None) -> None:
        self._random_state = random_state
        self._results: list[dict[str, Any]] = []
        self._rng = random.Random(random_state)

    def grid_search(
        self,
        param_grid: dict[str, list[Any]],
        objective_fn: ObjectiveFn,
        maximize: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        combinations = self._generate_combinations(param_grid)
        logger.info(f"Grid search: {len(combinations)} combinaciones")

        best = None
        best_score = float("-inf") if maximize else float("inf")
        evaluated: list[dict[str, Any]] = []

        for params in combinations:
            try:
                score = objective_fn(**params, **kwargs)
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error evaluando {params}: {e}")
                continue
            evaluated.append({"params": params, "score": score})
            if maximize:
                if score > best_score:
                    best_score, best = score, params
            else:
                if score < best_score:
                    best_score, best = score, params

        self._results.extend(evaluated)
        return {
            "method": "grid_search",
            "best_params": best,
            "best_score": best_score if best is not None else None,
            "total_evaluations": len(evaluated),
            "results": evaluated,
        }

    def random_search(
        self,
        param_space: dict[str, list[Any]],
        objective_fn: ObjectiveFn,
        n_iter: int = 50,
        maximize: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        logger.info(f"Random search: {n_iter} iteraciones")

        best = None
        best_score = float("-inf") if maximize else float("inf")
        evaluated: list[dict[str, Any]] = []

        for _ in range(n_iter):
            params = {key: self._rng.choice(values) for key, values in param_space.items()}
            try:
                score = objective_fn(**params, **kwargs)
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error evaluando {params}: {e}")
                continue
            evaluated.append({"params": params, "score": score})
            if maximize:
                if score > best_score:
                    best_score, best = score, params
            else:
                if score < best_score:
                    best_score, best = score, params

        self._results.extend(evaluated)
        return {
            "method": "random_search",
            "best_params": best,
            "best_score": best_score if best is not None else None,
            "total_evaluations": len(evaluated),
            "results": evaluated,
        }

    def bayesian_optimization(
        self,
        param_space: dict[str, list[Any]],
        objective_fn: ObjectiveFn,
        n_trials: int = 50,
        maximize: bool = True,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Optimización bayesiana usando Optuna (TPE sampler)."""
        try:
            import optuna
        except ImportError as exc:  # pragma: no cover
            raise ImportError("optuna es necesario para la optimización bayesiana") from exc

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        direction = "maximize" if maximize else "minimize"

        def suggest(trial: Any) -> dict[str, Any]:
            params: dict[str, Any] = {}
            for key, values in param_space.items():
                params[key] = trial.suggest_categorical(key, list(values))
            return params

        def objective(trial: Any) -> float:
            params = suggest(trial)
            try:
                return objective_fn(**params, **kwargs)
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error evaluando {params}: {e}")
                raise optuna.TrialPruned() from e

        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(
            seed=self._random_state
        ))
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        evaluated = [
            {"params": trial.params, "score": trial.value}
            for trial in study.trials
            if trial.value is not None
        ]
        self._results.extend(evaluated)
        return {
            "method": "bayesian_optimization",
            "best_params": study.best_params if evaluated else None,
            "best_score": study.best_value if evaluated else None,
            "total_evaluations": len(evaluated),
            "results": evaluated,
        }

    def _generate_combinations(self, param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
        if not param_grid:
            return [{}]
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def get_results(self) -> list[dict[str, Any]]:
        return self._results.copy()

    def clear_results(self) -> None:
        self._results.clear()
