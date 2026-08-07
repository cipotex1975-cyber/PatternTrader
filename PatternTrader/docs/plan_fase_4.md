# Plan Fase 4 — Persistencia y API real

> **Fecha**: 2026-08-07
> **Estado**: COMPLETADO (2026-08-07) — ver `docs/GAP_ANALYSIS.md` Fase 4
> **Base**: `docs/GAP_ANALYSIS.md` — Fase 4 (Persistencia y API real)

## Objetivo

Cerrar el gap de datos: pasar de runtime 100% en memoria a persistencia real (Postgres vía SQLAlchemy async + Alembic), conectar los engines por *write-through* de repositorios y exponer routers reales (`/models`, `/trades`, `/lifecycle`) compartiendo el estado del pipeline.

## Decisiones acordadas

- **Write-through por repositorios**: cada engine (`LifecycleEngine`, `SignalEngine`, `ExecutionEngine`) acepta un repositorio opcional y persiste en cada mutación; con `repository=None` queda 100% en memoria → los 147 tests actuales no requieren DB.
- **Postgres vía Docker Compose** para desarrollo y tests.
- **Alcance completo**: `patterns`+`lifecycles`, `signals`, `trades`, `backtests`, `knowledge_entries`, `predictions`, `ml_models`, `metrics`, `logs`.

---

## M1 — Infraestructura de DB

1. **Override `DATABASE_URL`**: en `app/core/config/settings.py:23-27`, la property `DatabaseSettings.url` debe dar prioridad a `os.environ.get("DATABASE_URL")`. Test unitario en `tests/unit/test_config.py`.
2. **`app/database/base.py`**: añadir `init_db()` (`Base.metadata.create_all` vía `conn.run_sync`), `reset_engine()` (limpiar singletons `_engine`/`_session_factory` para tests), `pool_pre_ping=True`; exportar `get_engine`/`get_session_factory`/`init_db`/`reset_engine` en `app/database/__init__.py`.
3. **Alembic**: crear `alembic.ini`, `migrations/env.py` async (lee URL desde `get_settings()` con override, ejecuta vía `run_sync`), migración inicial autogenerada, y documentar `alembic upgrade head`.
4. **Docker Compose**: `docker-compose.yml` con `postgres:16` + env `DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD` (compatibles con `config/settings.yaml:13-21`).
5. **Lifespan** (`app/api/main.py`): `await init_db()` antes de arrancar los servicios.

## M2 — Repositorios por dominio (`app/database/repositories/`)

Mismo estilo que `KnowledgeRepository` (`app/learning/repository.py`): statements async, `_to_orm`/`_to_model`, `get_async_session`. Cada repositorio con variante en memoria para tests de engines.

- `AssetRepository.get_or_create(symbol)` — necesario por la FK `patterns.asset_id`.
- `LifecycleRepository`: `register_pattern(pattern)` → inserta `patterns` + `lifecycles`; `update_transition(...)` → actualiza `current_state` + JSON `transitions` + `closed_at`.
- `SignalRepository`: `add`, `update_status`, `list(filters)`, `get(signal_uuid)`.
- `TradeRepository`: `add`, `update_closed`, `list(status, symbol)`, `get(trade_uuid)`.
- `BacktestRepository`: `add(BacktestResult)` (config/metrics como JSON + columnas resumen), `list`, `get`.
- `PredictionRepository`: `add(prediction)`.
- `MLModelRepository`: `upsert(name, ...)`, `list`, `get(name)`.
- `MetricRepository`: registro de métricas puntuales.
- `LogRepository`: escritura de WARNING/ERROR a la tabla `logs`.

## M3 — Cablear write-through en runtime

- `LifecycleEngine(repository=None)` (`app/lifecycle/engine.py`): persistir en `register()` y en `transition()`/`add_transition`.
- `SignalEngine(repository=None)` (`app/signals/engine.py`): persistir en `create_signal` tras el dedup; `mark_delivered`/`mark_failed` actualizan estado.
- `ExecutionEngine(repository=None)` (`app/execution/engine.py`): `open_trade` → `add`; `close_trade` → `update_closed`.
- `PatternService` (`app/patterns/service.py`): construir los repos y pasarlos al pipeline, al signal engine y al execution engine.
- Lifespan: `LearningService` pasa a `KnowledgeRepository` (DB) en producción; tests siguen con `MemoryKnowledgeRepository`.
- Persistir `predictions` en los endpoints de predict y `ml_models` al entrenar/cargar modelos (`train_offline`, `ScoringEngine`); `metrics`/`logs` vía repositorios.

## M4 — API: estado compartido + routers reales

- Dependencias FastAPI (`get_learning_service`, `get_pattern_service`) con las instancias reales en `app.state` desde el lifespan.
- `dashboard.py`: usar el `LifecycleEngine` compartido del pipeline + `pipeline.stats()` + trades del `ExecutionEngine` (eliminar la instancia vacía propia de `app/api/routes/dashboard.py:8`).
- `signals.py`: leer/escribir desde `SignalRepository` (reemplaza `_signals=[]` de `app/api/routes/signals.py:9`).
- `learning.py`: usar el `LearningService` compartido (eliminar instancia propia de `app/api/routes/learning.py:11`).
- **Routers nuevos**:
  - `/api/v1/trades`: GET `/` con filtros open/closed/symbol, GET `/{id}`.
  - `/api/v1/lifecycle`: GET `/statistics`, GET `/`, GET `/{id}`, GET `/pattern/{pattern_id}`.
  - `/api/v1/models`: GET `/` desde `MLModelFactory.get_all()` + `ml_models`, GET `/{name}`, POST `/{name}/predict`.
- Registrar los routers nuevos en `app/api/main.py` bajo `/api/v1`.
- `backtests.py`: persistir resultados en `BacktestRepository` y servir GET desde DB (IDs = PK int); se mantienen los generadores sintéticos como input por diseño.

## M5 — Calidad y compatibilidad

- `repository=None` por defecto → tests unitarios actuales sin DB (retrocompat).
- Flake8 + mypy limpio (`--python-version 3.12` por los stubs de numpy) en archivos nuevos/modificados.
- Actualizar `docs/GAP_ANALYSIS.md` (marcar Fase 4 ✅) y `docs/INSTALLATION.md`/`docs/CONFIGURATION.md` (DATABASE_URL, alembic, docker-compose).

## M6 — Tests

- `tests/unit/test_database.py`: override `DATABASE_URL`, `init_db`/`reset_engine`.
- `tests/unit/test_repositories.py`: CRUD de cada repositorio nuevo contra la DB de test (Postgres de compose).
- Extensiones de `tests/unit/test_lifecycle.py` / `tests/unit/test_execution.py` / `tests/unit/test_pipeline.py`: con repos inyectados (memoria/sqlite) verificando que la persistencia acompaña a la transición.
- `tests/unit/test_api_*.py`: routers nuevos/refactorizados con `app.dependency_overrides` y stub del provider (sin red).

---

## Riesgos / notas

- `get_async_session` hace commit al salir (`app/database/base.py:48`) → doble commit en `clear()`: aceptable, no se toca.
- `get_engine` es singleton lazy → `reset_engine()` obligatorio en tests que cambian la URL.
- El lifespan conecta al provider Binance: los tests de API deben sobrescribir dependencias para evitar red.
- `patterns` requiere `asset_id` (FK) → resolver vía `AssetRepository.get_or_create`.

## Archivos relevantes

- `app/database/base.py`, `app/database/models.py`, `app/database/__init__.py`
- `app/core/config/settings.py` (`DatabaseSettings.url`)
- `app/learning/repository.py` (patrón de repositorio a replicar)
- `app/lifecycle/engine.py`, `app/signals/engine.py`, `app/execution/engine.py`, `app/patterns/service.py`
- `app/api/main.py`, `app/api/routes/{dashboard,signals,learning,backtests}.py`
- `app/ml/factory.py` (`MLModelFactory`), `app/ml/base.py` (`MLPrediction`)
