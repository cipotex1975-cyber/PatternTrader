from app.backtesting.optimization import BacktestOptimizer


def _objective(**params):
    return -((params.get("a", 0) - 3) ** 2) - ((params.get("b", 0) - 5) ** 2)


def test_grid_search_finds_best():
    optimizer = BacktestOptimizer(random_state=42)
    result = optimizer.grid_search(
        {"a": [1, 2, 3, 4], "b": [4, 5, 6]}, _objective
    )
    assert result["best_params"] == {"a": 3, "b": 5}
    assert result["total_evaluations"] == 12
    assert result["best_score"] == 0.0


def test_grid_search_minimization():
    optimizer = BacktestOptimizer(random_state=42)
    result = optimizer.grid_search(
        {"a": [0, 3]}, _objective, maximize=False
    )
    assert result["best_params"] == {"a": 0}


def test_random_search_finds_best():
    optimizer = BacktestOptimizer(random_state=42)
    result = optimizer.random_search(
        {"a": list(range(0, 10)), "b": list(range(0, 10))},
        _objective,
        n_iter=50,
    )
    assert result["total_evaluations"] == 50
    assert result["best_score"] > -5


def test_random_search_seeded_is_reproducible():
    r1 = BacktestOptimizer(random_state=7).random_search(
        {"a": list(range(100))}, _objective, n_iter=20
    )
    r2 = BacktestOptimizer(random_state=7).random_search(
        {"a": list(range(100))}, _objective, n_iter=20
    )
    assert r1["best_score"] == r2["best_score"]


def test_bayesian_optimization():
    optimizer = BacktestOptimizer(random_state=42)
    result = optimizer.bayesian_optimization(
        {"a": [0, 1, 2, 3, 4, 5], "b": [3, 4, 5, 6, 7]},
        _objective,
        n_trials=20,
    )
    assert result["method"] == "bayesian_optimization"
    assert result["total_evaluations"] == 20
    assert result["best_params"]["a"] == 3
    assert result["best_params"]["b"] == 5


def test_results_tracking_and_clear():
    optimizer = BacktestOptimizer()
    optimizer.grid_search({"a": [1, 2]}, _objective)
    assert len(optimizer.get_results()) == 2
    optimizer.clear_results()
    assert optimizer.get_results() == []
