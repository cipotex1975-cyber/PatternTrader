# PatternTrader - Agent Instructions

## Commands & Verification
- **Run all unit tests**: `pytest`
- **Run specific tests**: `pytest tests/unit/test_ml_training.py`
- **Type checking**: `mypy app/`
- **Linting**: `flake8 app/ tests/`
- **Formatting**: `black app/ tests/ && isort app/ tests/`
- **Database Migrations**: `alembic upgrade head`
- **Train & Compare ML Models**: `python train_and_compare.py app/datos_test/USDCAD_H1_201005311000_202606010000.txt --model all --metric roc_auc`
- **Run Backtest**: `python run_backtest.py`
- **Start API Server**: `python -m app.main`

## Verification Workflow
When making code changes, run verification in this order:
1. `black app/ tests/ && isort app/ tests/`
2. `mypy app/`
3. `flake8 app/ tests/`
4. `pytest`

## Architecture & Layout
- **Clean Architecture / DDD**: Organized across `app/core/`, `app/market/`, `app/patterns/`, `app/lifecycle/`, `app/scoring/`, `app/confirmation/`, `app/risk/`, `app/signals/`, `app/strategy/`, `app/ml/`, `app/backtesting/`, `app/execution/`, `app/database/`, `app/api/`.
- **ML Training & Per-Pair Selection**: `train_and_compare.py` trains all 9 models, compares metrics, saves the winner as `{model_name}_{symbol}.{ext}` in `models/` with a corresponding `.meta.json` sidecar.
- **Scoring & ML Integration**: `ScoringEngine` (`app/scoring/engine.py`) automatically loads per-symbol models via sidecar rehydration.
- **ML Factory**: Use `MLModelFactory.create("name")` for cached singletons (API) and `MLModelFactory.create_new("name", **kwargs)` for independent instances (training).

## Revisión de documentación
- revisar documentacion ubicada en docs/
