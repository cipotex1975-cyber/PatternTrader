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
- **ML Training & Per-Pair Selection**: `train_and_compare.py` trains all 9 models, compares metrics, saves the winner as `{model_name}_{symbol}_{tf}.{ext}` in `models/` with a corresponding `.meta.json` sidecar. The model is indexed by **symbol + timeframe**; timeframes are canonicalized by `normalize_timeframe` (`app/core/constants/market.py`), so `H1` and `1h` are the same key.
- **Scoring & ML Integration**: `ScoringEngine` (`app/scoring/engine.py`) loads models via sidecar rehydration with **strict** resolution: `_load_ml_model_for_key(symbol, timeframe)` requires an exact `timeframe` match, and a sidecar without that key is ignored. Falls back to the generic `*.pkl` with no sidecar. Filename convention is centralized in `app/ml/naming.py` — use it from both the save and load sides so the two cannot drift.
- **ML Factory**: Use `MLModelFactory.create("name")` for cached singletons (API) and `MLModelFactory.create_new("name", **kwargs)` for independent instances (training).
- **Data Source Separation**: Signals carry `data_source`. Live API server (`python -m app.main`) tags signals `data_source="live"` (Yahoo Finance/Binance); `simulate_pipeline.py` tags them `data_source="simulation"` (historical files). The API `GET /api/v1/signals/` filters to `live` by default; use `?data_source=all|simulation` for others. Never mix historical-file data into the live pipeline.
- **Timeframes**: `market.default_timeframes` in `config/settings.yaml` is the single source of truth for the pipeline; `PatternService` (`app/patterns/service.py`) iterates exactly that list to create one scheduler task per symbol × timeframe. Do not add a second timeframe list under `patterns.lifecycle`. Startup logs the active list (`PatternService started: N symbols x M timeframes [...]`).

## Revisión de documentación
- revisar documentacion ubicada en docs/
