from __future__ import annotations

from typing import Any, Callable, Optional

from app.core.logger import get_logger

logger = get_logger("OptimizerEngine")


class OptimizerEngine:
    def __init__(self) -> None:
        self._results: list[dict[str, Any]] = []

    async def grid_search(
        self,
        param_grid: dict[str, list[Any]],
        evaluate_fn: Callable[..., float],
        **kwargs: Any,
    ) -> dict[str, Any]:
        logger.info(f"Starting grid search with {self._count_combinations(param_grid)} combinations")

        best_score = float("-inf")
        best_params = {}

        combinations = self._generate_combinations(param_grid)
        for params in combinations:
            try:
                score = evaluate_fn(**params, **kwargs)
                self._results.append({"params": params, "score": score})

                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception as e:
                logger.error(f"Error evaluating params {params}: {e}")

        return {
            "best_params": best_params,
            "best_score": best_score,
            "total_evaluations": len(combinations),
            "results": self._results,
        }

    async def random_search(
        self,
        param_distributions: dict[str, list[Any]],
        evaluate_fn: Callable[..., float],
        n_iter: int = 100,
        **kwargs: Any,
    ) -> dict[str, Any]:
        import random

        logger.info(f"Starting random search with {n_iter} iterations")

        best_score = float("-inf")
        best_params = {}

        for _ in range(n_iter):
            params = {
                key: random.choice(values)
                for key, values in param_distributions.items()
            }

            try:
                score = evaluate_fn(**params, **kwargs)
                self._results.append({"params": params, "score": score})

                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception as e:
                logger.error(f"Error evaluating params {params}: {e}")

        return {
            "best_params": best_params,
            "best_score": best_score,
            "total_evaluations": n_iter,
            "results": self._results,
        }

    def _generate_combinations(self, param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
        if not param_grid:
            return [{}]

        keys = list(param_grid.keys())
        values = list(param_grid.values())

        combinations = []
        self._recursive_combinations(keys, values, 0, {}, combinations)
        return combinations

    def _recursive_combinations(
        self,
        keys: list[str],
        values: list[list[Any]],
        index: int,
        current: dict[str, Any],
        result: list[dict[str, Any]],
    ) -> None:
        if index == len(keys):
            result.append(current.copy())
            return

        for value in values[index]:
            current[keys[index]] = value
            self._recursive_combinations(keys, values, index + 1, current, result)

    def _count_combinations(self, param_grid: dict[str, list[Any]]) -> int:
        count = 1
        for values in param_grid.values():
            count *= len(values)
        return count

    def get_results(self) -> list[dict[str, Any]]:
        return self._results.copy()

    def clear_results(self) -> None:
        self._results.clear()
