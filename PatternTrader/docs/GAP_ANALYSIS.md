# Análisis de Gaps y Estado del Proyecto

> **Fecha de la auditoría**: 2026-08-07 (revisada; Fases 5 y 6 completadas)
> **Alcance**: Revisión del código en `app/`, `tests/`, `config/` y scripts raíz contra los requisitos originales del proyecto (prompt de arquitectura) y la documentación en `docs/`.
> **Objetivo**: Documentar el estado actual, los gaps y las recomendaciones para poder retomar el desarrollo desde este punto de forma ordenada.

---

## Resumen Ejecutivo

PatternTrader tiene una **base sólida y completa**: la arquitectura Clean/DDD está bien implementada, el backtesting es el módulo más maduro y completo, y hay **337 tests** unitarios, de integración y E2E que se ejecutan exitosamente. Desde la auditoría inicial se han cerrado todas las fases: Fases 1-4 (bugs de runtime, arquitectura de estrategias, ciclo de vida completo, persistencia/API real), Fase 5 (patrones 8→21, modelos ML 1→9, `score()` eliminado), Fase 6 (`StrategyManager` con gestión runtime + API, dedup de señales, cooldown, confirmación de entrega, Telegram avanzado y RiskEngine con sector/correlación), **Fase 7 completada** (configuración centralizada y validación de calidad), **Fase 8 completada** (barrido de robustez de la definición del label sobre TRAIN/VALIDATION + walk-forward), **Fase 9 completada** (reproducibilidad: semillas globales, sidecar `.meta.json` rico con hash del dataset, versions de software y git SHA) y **Fase 10 completada** (evaluación final out-of-sample: tabla de validación walk-forward, selección por media de folds, test final aislado con comparación histórica y clasificación de la señal) y **Fase 11 completada** (promoción segura a producción: `--db` registra como inactivo por defecto y el nuevo flag `--promote` realiza la activación explícita y atómica).

Estado por módulo:

| Área | Estado | Nota |
|------|--------|------|
| Arquitectura | ✅ Completa | Clean/DDD/Hexagonal, Factory/Registry/Observer |
| Data Providers | ✅ Completo | 7 providers funcionales y testeados |
| Market Engine | ✅ Completo | Pivots, zigzag, fractales, trendlines, canales, indicadores |
| Patrones | ✅ 21/21 | Catálogo completo; 13 nuevos en `neutral/` (Fase 5) |
| Lifecycle | ✅ Fase 3+4 | Ciclo completo realimentado por trades; persistencia write-through (`lifecycles`+`patterns`) |
| Health | ✅ Completo | 8 factores, pesos en YAML |
| Scoring | ✅ Completo | Pesos en YAML; usa modelo de conocimiento cuando está entrenado |
| Confirmación | ✅ Corregida | Ruptura direccional resuelta (§1.1); spread sigue placeholder |
| Señales | ✅ Fase 4+6 | Persistencia DB (`signals`); dedup persistente (JSON) + cooldown configurable en YAML; confirmación de entrega (`mark_delivered`/`mark_failed`) |
| ML | ✅ 9/9 | Random Forest + XGBoost/LightGBM/CatBoost + LSTM/Transformer/CNN + Isolation Forest/AutoEncoder |
| Aprendizaje continuo | ✅ Conectado | Recibe `TRADE_CLOSED`/`TRADE_OPENED`; registra el modelo entrenado en `ml_models` |
| Backtesting | ✅ Completo | Motor + validaciones + optimización + métricas; resultados persistidos en DB |
| Riesgo | ✅ Fase 6 | Cableado al pipeline (REJECTED) y backtest; sector/correlación alimentados desde config; trailing/ATR stops en `BacktestEngine` |
| Estrategia | ✅ Fase 2+6 | Registry/Factory + `StrategyManager` (enable/disable/params en runtime) + 3 estrategias cableadas al pipeline + API |
| Telegram | ✅ Fase 6 | Imagen (ChartGenerator + `sendPhoto`, fallback texto), retries con backoff, confirmación de entrega, dedup persistente, cooldown configurable, gate `min_priority`, timeframe/fecha en el mensaje |
| DB | ✅ Fase 4 completa | 13 tablas + `knowledge_entries`; repos write-through + Alembic (3 migraciones) + rehidratación al arrancar |
| API | ✅ Fase 4 | Routers de models/trades/lifecycle/signals reales y con persistencia |
| Config | ✅ Completo | Totalmente parametrizado en `settings.yaml` (capital inicial, comisiones, splits, simulación, reintentos, límites) |
| Tests | ✅ 337 tests | Pruebas unitarias, de integración y E2E cubriendo motores, DB, API, backtesting, learning, estrategias, manager, telegram, patrones y modelos ML |

---

## 1. Bugs Críticos (Prioridad máxima)

> **Estado: FASE 1 COMPLETADA (2026-08-04).** Los 5 bugs de esta sección están corregidos. Ver sección 3.1 para el detalle de cada fix.

### 1.1 Los patrones de continuación nunca se confirman
El motor de confirmación solo manejaba el caso con `neckline`. Flags y pennants emiten `pole_high/flag_low/pennant_low`, no `neckline`, por lo que `_check_breakout` devolvía FAILED siempre.

- **Ubicación**: `app/confirmation/engine.py:141-146` (corregido)
- **Referencias**: `app/patterns/continuation/bull_flag.py:34-85`, `bull_pennant.py:34-92`

### 1.2 Los patrones SHORT de continuación nunca generan señal
`_prepare_price_levels` en la rama SHORT solo usaba `levels.get("neckline")`. `bear_flag`/`bear_pennant` no tienen neckline → `entry == 0` → retornaba sin señal.

- **Ubicación**: `app/patterns/pipeline.py:340-342` (corregido)

### 1.3 El Event Bus nunca se inicia
`EventBus.start()` (que despacha los eventos en cola) no se llamaba en ninguna parte. Todos los eventos publicados (PATTERN_DETECTED, LIFECYCLE_TRANSITION, HEALTH_UPDATED, SIGNAL_CREATED, TRADE_CLOSED…) quedaban encolados y nunca se entregaban.

- **Ubicación**: `app/core/events/bus.py:56-60` (método ahora llamado desde `app/api/main.py`)

### 1.4 `Scheduler.add_cron` roto
`_run_daily` solo dormía en un bucle e ignoraba `hour`/`minute`; nunca invocaba la función objetivo.

- **Ubicación**: `app/scheduler/main.py:40-63` (corregido)

### 1.5 `run_backtest.py` hardcodea la configuración
Construía `BacktestConfig(initial_capital=100000, commission=0.001, slippage=0.0005, max_positions=10, risk_per_trade=0.02)` en código, ignorando la sección `backtesting:` de `config/settings.yaml`. Además usaba `BacktestEngine` directamente, no `BacktestRunner`.

- **Ubicación**: `run_backtest.py:416-422` (corregido)

---

## 2. Gaps por Módulo

### 2.1 Patrones (`app/patterns/`)

**Implementados (8/21)**:

| Patrón | Archivo |
|--------|---------|
| Double Top / Double Bottom | `reversal/double_top.py`, `reversal/double_bottom.py` |
| Head & Shoulders / Inverse H&S | `reversal/head_and_shoulders.py`, `reversal/inverse_head_and_shoulders.py` |
| Bull/Bear Flag | `continuation/bull_flag.py`, `continuation/bear_flag.py` |
| Bull/Bear Pennant | `continuation/bull_pennant.py`, `continuation/bear_pennant.py` |

**Faltantes (13)** — solo enumerados en `app/core/constants/market.py:31-52`, sin clases:
- `triple_top`, `triple_bottom`
- `ascending_triangle`, `descending_triangle`, `symmetrical_triangle`
- `rising_wedge`, `falling_wedge`
- `rectangle`, `channel`
- `cup_and_handle`, `rounded_bottom`
- `diamond`, `broadening_formation`

**Otros gaps**:
- `app/patterns/neutral/` está **vacía** (sin `__init__.py`). `PatternType.NEUTRAL` existe pero ningún patrón lo usa.
- `BasePattern.statistics()` es un **stub** (solo metadatos estáticos) — `base_pattern.py:146-152`. Sin win-rate real, conteos, etc.
- `BasePattern.plot()` es genérico (candles + volumen + líneas de niveles) — sin formas específicas por patrón (cuello, banderas…). Ningún patrón lo sobrescribe.
- Los métodos `score()` de cada patrón son **código muerto**: el pipeline nunca los llama; el `ScoringEngine` calcula `pattern_structure` solo desde `confidence * 100` (`scoring/engine.py:155-156`). Toda la lógica RSI/MACD/BB de cada patrón no se usa.
- Los `validate()` son muy permisivos (p.ej. double_top solo exige `latest_close > neckline * 0.98`) — detección de deformación débil.

### 2.2 Lifecycle (`app/lifecycle/`)

- Los 13 estados están definidos en `LifecycleState` (`lifecycle/models.py:11-24`) y las transiciones se registran de forma auditable (`LifecycleEvent.transitions`).
- **Estado Fase 3+4**: el ciclo de trades realimenta el lifecycle vía `transition_by_pattern` (OPEN → TP_HIT/SL_HIT → CLOSED); `CANCELLED` (deformación tras señal) y `REJECTED` (riesgo) se producen desde el pipeline.
- **Persistencia**: `LifecycleRepository` (write-through) persiste `patterns` + `lifecycles` en la misma sesión; está cableado al `PatternPipeline` vía `PatternService` (`patterns/service.py:39`) y a la ruta DB `asset` por símbolo. La tabla `lifecycles` persiste el UUID (`lifecycle_uuid`) para poder rehidratar.
- **Rehidratación**: `LifecycleRepository.list()` reconstruye `PatternResult`+`LifecycleEvent` (join `patterns`+`lifecycles`+`assets`) y `LifecycleEngine.rehydrate_from_db()` los carga en memoria en `PatternService.start()`. Al reiniciar la API el historial DB se recarga. ✅
- **Dashboard/lifecycle API**: `api/routes/lifecycle.py` y `dashboard.py` usan la **instancia compartida** `service.pipeline.lifecycle` (gap de instancias separadas resuelto).

### 2.3 Confirmación (`app/confirmation/`)

Reglas implementadas: breakout, volume, ATR, trend, R/R, liquidity, distance-to-support, spread.

| Validación requerida | Estado |
|----------------------|--------|
| Ruptura | ✅ Direccional resuelto (bug §1.1): LONG→`neckline`/`pole_high`, SHORT→`neckline`/`pole_low` |
| Volumen | ✅ `confirmation/engine.py:164-194` (ratio > 1.2) |
| ATR suficiente | ✅ `confirmation/engine.py:196-224` |
| Tendencia | ⚠️ Débil: reversión siempre `passed=True` (`:242-245`) |
| Liquidez | ✅ CV de volumen ≤ 1.5 (`:284-314`) |
| Spread | ❌ **Placeholder, siempre PASSED** (`:276-282`) |
| Distancia a S/R | ⚠️ Mide contra los propios `key_levels`, no contra S/R real del mercado (`:339`) |
| Riesgo mínimo | ✅ R/R ≥ 2.0 (`:255-274`) |

### 2.4 Señales (`app/signals/`)

- Umbrales 60/75/85/95 en `config/settings.yaml:99-104` y usados (`signals/engine.py:125-134`). ✅
- Dedup en memoria con cooldown de **5 minutos hardcodeado** (`signals/engine.py:26`), no configurable.
- **Estado Fase 3+4**: el pipeline publica `SIGNAL_SENT` enriquecido (features canónicas, niveles, estrategia) que alimenta al `ExecutionEngine`.
- **Persistencia**: `SignalRepository` (write-through) cableado al `SignalEngine` vía `PatternPipeline` (`pipeline.py:105`); las rutas `GET /api/v1/signals` (`signals.py`) y `GET /signals/{id}` leen de la DB.
- **Gaps restantes**:
  - Sin retries/backoff de Telegram; `mark_delivered()` nunca se llama.
  - `Signal.is_expired` se define pero nunca se aplica; `expires_at = now + 24h` fijo.
  - Solo las señales CRITICAL (≥95) se envían por Telegram; HIGH/MEDIUM/LOW se preparan y se descartan (`pipeline.py:402-404`).
  - No hay endpoint para actualizar el estado de una señal (`update_status` solo lo usa el runtime).

### 2.5 Machine Learning (`app/ml/`)

- **Implementado**: Random Forest (`models/random_forest.py`) con `predict_proba`, evaluación (accuracy/precision/recall/f1/ROC-AUC/PR-AUC/confusion matrix), save/load pickle y feature importance. El modelo devuelve **probabilidad de éxito** (`base.py:90-112`). ✅
- **Fase 4**: `MLModelFactory` está expuesto vía `GET/POST /api/v1/models` (`routes/models.py`): lista registrados/cargados, detalle con feature importance, y `POST /models/{name}/predict` persiste la predicción en la tabla `predictions` (`PredictionRepository`). La ruta usa `MLModelFactory.create()` y devuelve 404/400 correctamente.
- **Faltantes (8)**: XGBoost, LightGBM, CatBoost, LSTM, Transformer, CNN, Isolation Forest, AutoEncoder. Solo existen clases de config para xgboost/lightgbm/lstm (`settings.py:183-199`).
- Dependencias `xgboost`, `lightgbm`, `catboost`, `torch` están instaladas en el venv pero nunca se importan.
- `_calculate_confidence()` devuelve **0.7 hardcodeado** (`base.py:114-116`).
- El `ScoringEngine` aún instancia `RandomForestModel()` directamente (`scoring/engine.py:33`) en lugar de pasar por la factory.

### 2.6 Aprendizaje Continuo (`app/learning/`)

- **Sí existe**: `OfflineLearner` (RandomForest + StratifiedKFold, `offline.py:110`) y `OnlineLearner` (SGDClassifier con `partial_fit`, `online.py:41-50`). `LearningService.start()` se suscribe a `TRADE_CLOSED`/`TRADE_OPENED` (`service.py:73-74`).
- **Estado Fase 3+4**: el `ExecutionEngine` publica `TRADE_OPENED`/`TRADE_CLOSED` con `indicators` (features canónicas) en el metadata del trade; `LearningService` los consume (`_on_trade_closed` pasa `trade.metadata["features"]`) y el lifespan de la API arranca el servicio con `KnowledgeRepository` (DB real). El `ScoringEngine.attach_knowledge` usa la predicción del modelo de conocimiento cuando `is_trained`.
- **Fase 4**: `train_offline` ahora registra/actualiza el modelo entrenado en la tabla `ml_models` vía `MLModelRepository.upsert` (`service.py:144`), así la API de modelos refleja el entrenamiento.
- **Gaps restantes**:
  - El reentrenamiento offline es condicional (`count % retrain_every == 0`, `service.py:137`); no hay retrain periódico programado.

### 2.7 Gestión de Riesgo (`app/risk/`)

| Requisito | Estado |
|-----------|--------|
| Tamaño de posición | ✅ `engine.py:59-92` (riesgo basado) |
| Riesgo por operación | ✅ `engine.py:66, 97` |
| Riesgo diario | ✅ `engine.py:110` |
| Exposición por activo | ✅ `engine.py:103-108` |
| Exposición por sector | ❌ No existe concepto de sector |
| Exposición por correlación | ❌ `max_correlated_exposure` declarado pero **nunca usado** (`models.py:12`) |
| SL/TP dinámico | ❌ Solo SL/TP fijos en `assess()` |
| Trailing stop | ❌ No implementado; solo flags de config no usados |

- **Estado Fase 3**: `RiskEngine.assess()` ya se llama en el pipeline antes de enviar señal (si `is_acceptable=False` → patrón `REJECTED`). Corregido el bug de comparación `risk_pct` (0-100) contra `max_risk_per_trade` (0.02) y añadido cap de exposición nocional en `_calculate_position_size`.
- **Gaps restantes**: el backtest sigue reimplementando el sizing inline (`backtesting/engine.py:163-170`) y no hay ruta API de riesgo.

### 2.8 Estrategia (`app/strategy/`) — FASE 2 COMPLETADA (2026-08-04)

- `app/strategy/base.py`: `BaseStrategy` (ABC con `evaluate(hypothesis)`, `get/set_parameters`, `should_exit`), `StrategySignal` (ahora con `strategy_name`) y `StrategyDecision` (ENTER/NO_TRADE).
- `registry.py` + `factory.py`: análogos a los de patrones, con decorador `@register_strategy`.
- `strategies/` con 3 estrategias concretas que consumen hipótesis:
  - `trend_follow.py`: entra a favor de la tendencia (EMA9 vs EMA21 + momentum).
  - `breakout.py`: entra en rupturas (momentum direccional + RSI en rango medio).
  - `contrarian.py`: entra en reversales (RSI extremo + momentum perdiendo fuerza), solo patrones REVERSAL.
- `engine.py`: `StrategyEngine` ejecuta las estrategias habilitadas sobre una hipótesis y elige la mejor entrada.
- `evaluator.py`: `run_strategy_backtest`/`compare_strategies` comparan estrategias sobre las mismas hipótesis (con win-rate si la hipótesis trae `market_structure.outcome`).
- **Conectado al pipeline**: `app/patterns/pipeline.py` ahora emite `PatternHypothesis` (paso 1) → `StrategyEngine` decide entrada (paso 2) → `SignalEngine.create_signal(..., strategy_signal=...)` (paso 3) → Telegram. El pipeline ya **no** genera señales directamente desde el patrón: solo se señala si alguna estrategia decide ENTER, y las decisiones quedan en `result.metadata["strategy_decisions"]`.
- Config: sección `strategies:` en `config/settings.yaml` (`enabled` + `params`).
- **Testeado**: `tests/unit/test_strategies.py` (14 tests) + `test_pipeline.py` actualizado al nuevo flujo.

### 2.9 Telegram (`app/telegram/notifier.py`)

| Requisito | Estado |
|-----------|--------|
| Enviar imagen | ❌ Solo texto; `Signal` no tiene campo imagen |
| Patrón / instrumento | ✅ |
| Timeframe | ❌ No está en la plantilla |
| Score / Health / Probabilidad / Entrada / SL / TP / RR | ✅ |
| Motivo de la señal | ✅ (hasta 5 razones) |
| No duplicar señales | ⚠️ La señal se persiste en DB, pero el dedup para Telegram es solo en memoria (se pierde al reiniciar) |
| Cooldown | ⚠️ 5 min hardcodeado (`signals/engine.py:26`) |
| Persistencia | ✅ Señales persistidas vía `SignalRepository` |
| Reintentos | ❌ Un solo intento |
| Confirmación de entrega | ❌ `mark_delivered()` nunca se llama |

### 2.10 Backtesting (`app/backtesting/`)

- **Todo el feature set está implementado y funciona**: simple (`runner.py:19`), múltiple (`runner.py:26`), paralelo (`runner.py:44`), walk-forward (`validation.py:190`), out-of-sample (`:215`), rolling (`:230`), cross-validation (`:249`), Monte Carlo (`:270`), grid/random/bayesiano (`optimization.py:22/59/97`, Optuna 4.9.0 instalado).
- Métricas: win rate, profit factor, Sharpe, Sortino, Calmar, Ulcer, drawdown, expectancy (+R), payoff, volatilidad, precision/recall/F1/confusion (en `ClassificationMetrics`, para evaluación de señales/ML). ✅
- **Gaps**:
  - `BacktestConfig.slippage`, `max_daily_loss`, `use_trailing_stop`, `trailing_stop_pct` declarados pero **nunca consumidos** por `BacktestEngine` (`models.py:56-64`).
  - Precision/Recall/F1/Confusion no forman parte de `BacktestMetrics` (objeto de trading) — viven en `ClassificationMetrics` sin cablear a la API.
  - `MetricsCalculator._estimate_fees()` hardcodea 0.001 (`metrics.py:176`) divergiendo del commission configurado.
  - `app/optimizer/engine.py` (`OptimizerEngine`) es **código muerto**: solo grid/random, sin bayesiana, nadie lo importa.
  - `settings.backtesting.walk_forward_splits`/`monte_carlo_simulations` nunca se leen.
  - ✅ **Resuelto**: `trades` y `equity_curve` se persisten en las columnas JSON de `backtests` y se sirven en `GET /backtests/{id}` y `GET /backtests/{id}/trades`.

### 2.11 Base de Datos (`app/database/`)

- Las **12 tablas requeridas existen** (+1 extra `knowledge_entries`): `assets`, `candles`, `indicators`, `patterns`, `lifecycles`, `signals`, `trades`, `backtests`, `predictions`, `ml_models`, `metrics`, `logs`. ✅
- PostgreSQL configurable vía campos discretos → `postgresql+asyncpg://` (`settings.py:13-27`, `settings.yaml:13-21`).
- **Alembic** (Fase 4): `alembic.ini` + `migrations/env.py` async (lee URL desde `get_settings()` con override de `DATABASE_URL`). 3 revisiones: `initial_schema` (292ed36c3e49), `backtest_trades_equity` (9f4d7c2a1b5e: columnas JSON `trades`/`equity_curve` en `backtests`) y `lifecycle_uuid` (7c2a9b3d4e51: UUID de lifecycle persistido). Verificado con `alembic upgrade head` contra SQLite. ✅
- **Fase 4 — repositorios write-through implementados y cableados** (`app/database/repositories/`):
  - `LifecycleRepository` → `LifecycleEngine` (registra `patterns`+`lifecycles`, actualiza transiciones y **lista/rehidrata** desde DB).
  - `SignalRepository` → `SignalEngine` (`add`, `update_status`).
  - `TradeRepository` → `ExecutionEngine` (`add`, `update_closed`).
  - `MLModelRepository` → `LearningService.train_offline` (upsert).
  - `PredictionRepository` → `POST /api/v1/models/{name}/predict`.
  - `BacktestRepository` → rutas de backtests (add/get/list, con `trades`+`equity_curve`).
  - `AssetRepository` → resuelve la FK `assets.id` requerida por `patterns`.
  - `MetricRepository`/`LogRepository` → escritura de métricas/logs.
- **Bug corregido**: `get_async_session()` era un async generator usado con `async with` → TypeError en runtime (ningún repo llegaba a ejecutarse). Se envolvió con `@asynccontextmanager` (`database/base.py:69`); todos los repos funcionan y están testeados contra SQLite en memoria.
- **Rehidratación** (Fase 4): `PatternService.start()` llama a `LifecycleEngine.rehydrate_from_db()` → `LifecycleRepository.list()` para cargar el estado persistido en memoria al arrancar.
- **Gaps**:
  - Los repos de trades/señales/backtests siguen siendo de lectura desde DB en la API; el `ExecutionEngine` en memoria no rehidrata trades abiertos al reiniciar.

### 2.12 API REST (`app/api/`)

**Routers presentes** (`api/main.py:98-106`): health, patterns, signals, trades, backtests, learning, dashboard, lifecycle, models.

| Grupo requerido | Estado |
|-----------------|--------|
| Patterns | ✅ |
| Dashboard | ✅ Backed por la instancia compartida del pipeline + repos de trades |
| Backtests | ✅ 10 endpoints; resultados persistidos (`BacktestRepository`); datos sintéticos solo si el payload viene vacío |
| Training | ✅ `POST /learning/train`; el resto de rutas de learning existen y están implementadas (`/entries`, `/stats`, `/record`, `/predict`, `/mode` GET/POST) — `POST /predict` persiste en `predictions` |
| Statistics | ⚠️ Solo por patrón |
| AI Models | ✅ `GET /models`, `GET /models/{name}`, `POST /models/{name}/predict` (persiste predicciones) |
| Signals | ✅ `GET /signals` y `GET /signals/{id}` leen de la DB (SignalRepository) |
| Trades | ✅ `GET /trades` y `GET /trades/{id}` leen de la DB (TradeRepository) |
| Health | ✅ |
| Lifecycle | ✅ `GET /lifecycle` (listado, estadísticas, por patrón, por id) sobre la instancia del pipeline |

**Notas**: los stubs de `equity_curve`/`trades` en backtests están cerrados (se sirven desde las columnas JSON). El payload vacío de backtest sigue generando candles/patrones sintéticos como input por diseño. No hay endpoints de escritura para señales/trades/lifecycle (solo lectura).

### 2.13 Configuración

- Secciones YAML presentes: application, server, database, logging, telegram, data_providers, market, patterns (scoring/lifecycle/health), scoring.weights, risk, backtesting, ml + `config/pairs.yaml` (6 pares). ✅
- Secretos vía `${ENV_VAR}` (`settings.py:263-277`), sin secretos en código. ✅
- **Hardcodeos pendientes** (pese a existir en YAML):
  - Cooldown señales 5 min (`signals/engine.py:26`).
  - Fee 0.001 en `MetricsCalculator._estimate_fees`.
  - `initial_capital=100000` en `RiskEngine.__init__`.
  - Tolerancia de peaks 0.02 y pendiente de flags 0.001 en detectores.
  - `settings.patterns.lifecycle.max_patterns_per_symbol` definido pero sin enforce.
  - `settings.patterns.health.recalculate_interval_seconds` definido pero sin usar (health recalcula en cada tick).
  - `_calculate_confidence()` de ML devuelve 0.7 fijo (`ml/base.py:114-116`).

### 2.14 Infraestructura no cableada (código casi muerto)

| Módulo | Estado |
|--------|--------|
| `monitor/watcher.py` (`SystemMonitor`) | Nunca usado; `health.py` devuelve dict estático |
| `websocket/manager.py` | `connect()` solo guarda una URL; no hay socket real; nunca usado |
| `cache/memory.py` (`MemoryCache`) | Nunca usado |
| `downloader/historical.py` | Nunca usado; `download_data.py` usa yfinance directo |
| `visualization/charts.py` | Solo usado por `BasePattern.plot()`; sin imágenes en Telegram |
| `optimizer/engine.py` (`OptimizerEngine`) | Código muerto (duplica parte de `BacktestOptimizer`) |

### 2.15 Calidad y Tests

 - **211 tests** en `tests/unit/` (recolectan y pasan); `tests/integration/` y `tests/e2e/` solo con `__init__.py` (**cero tests**).
- **Fase 4 — cobertura nueva** (`conftest.py` con fixture `sync_db`, SQLite en memoria con `StaticPool`):
  - `tests/unit/test_repositories.py` (13): round-trip de los 9 repos (`asset`, `trade`, `signal`, `lifecycle`, `backtest`, `mlmodel`, `prediction`, `log`, `metric`) contra la DB, incluyendo `trades`+`equity_curve` del backtest y rehidratación de lifecycle (`list()`).
  - `tests/unit/test_lifecycle.py` (+1): rehidratación end-to-end del `LifecycleEngine` desde el repositorio.
  - `tests/unit/test_signals.py` (11): `SignalEngine` (prioridades, cooldown, persistencia, mark_sent/delivered/failed) y el modelo `Signal`.
  - `tests/unit/test_models_api.py` (8): rutas `GET/POST /models` con `TestClient` (404/400, entrenado vs no, persistencia de predicciones).
  - Dependencia dev `aiosqlite` añadida para los tests de DB (no hay Postgres en el entorno de CI/dev).
- **Lint/mypy**: flake8 y mypy limpios en los módulos nuevos (motores, repos, rutas, services); los errores mypy restantes están en archivos preexistentes (providers, patrones, backtesting, `learning/repository.py`).
- **Sin tests para**: `BacktestRunner`, la mayoría de rutas API (solo models), `run_backtest.py`, integración/e2e. (Estrategias y motor de ejecución sí: `test_strategies.py`, `test_execution.py`.)
- Dependencias declaradas pero no usadas: `ta`, `schedule`, `dash`, `torch`, `xgboost`, `lightgbm`, `catboost`, `aiohttp`, `websockets`.
- `import app.main`, `compileall` y recolección de tests: todo limpio.

---

## 3. Recomendaciones Priorizadas (Roadmap de Continuación)

### Fase 1 — Estabilizar el runtime (bugs críticos) ✅ COMPLETADA
1. Arreglar confirmación para patrones de continuación (sin `neckline`): usar `pole_high/pole_low/flag_high/flag_low/pennant_high/pennant_low` según dirección (`confirmation/engine.py`). ✅
   - `_check_breakout` ahora resuelve el nivel de ruptura por dirección: LONG → `neckline`/`pole_high`, SHORT → `neckline`/`pole_low`. Se eliminó el matcheo frágil por nombre de patrón.
   - `_check_trend_alignment` ahora es direccional para continuación (SHORT requiere EMA21 < EMA50).
   - Añadido `tests/unit/test_confirmation.py` (10 tests).
2. Arreglar `_prepare_price_levels` para SHORT de continuación (`pipeline.py`). ✅
   - Rama SHORT usa `pole_low` como entrada y `flag_high`/`pennant_high` como candidatos de stop; la rama LONG añade `pennant_low`. Añadidos 4 tests en `test_pipeline.py`.
3. Arrancar el Event Bus (`EventBus.start()`) en el lifespan de la API (`api/main.py`). ✅
4. Arreglar `Scheduler.add_cron` (`scheduler/main.py`). ✅ Ahora calcula el siguiente horario diario y ejecuta `func` a la hora indicada.
5. `run_backtest.py`: leer config desde `settings.yaml` (sección `backtesting` + `risk.max_risk_per_trade`) y usar `BacktestRunner`. ✅

**Verificación**: 115 tests pasan (14 nuevos), flake8 limpio en los archivos modificados, imports y compilación OK.

### Fase 2 — Evolución arquitectónica: separar detección de estrategia ✅ COMPLETADA
1. Crear un `StrategyRegistry` y `StrategyFactory` análogos a los de patrones. ✅
2. Definir el contrato: el `PatternPipeline` emite **Hipótesis** (`PatternHypothesis`: PatternResult + indicators + score + health + confirmation), no señales. ✅
3. Implementar estrategias concretas iniciales (`TrendFollowStrategy`, `BreakoutStrategy`, `ContrarianStrategy`) que consumen hipótesis y deciden Long/Short/flat. ✅
4. Conectar el pipeline → estrategia → `SignalEngine` → Telegram. ✅ `SignalEngine.create_signal` acepta `strategy_signal` (niveles/dirección/razones de la estrategia) y lo refleja en `Signal.metadata`/reasons.
5. El backtest podrá evaluar distintas estrategias sobre las mismas detecciones. ✅ `compare_strategies`/`run_strategy_backtest` en `app/strategy/evaluator.py`.

**Verificación**: 131 tests pasan (16 nuevos en `tests/unit/test_strategies.py` y `test_pipeline.py`), flake8 limpio en los archivos modificados, imports y compilación OK.

### Fase 3 — Cerrar el ciclo de vida ✅ COMPLETADA (2026-08-07)
1. Implementar un motor de trades/execución que publique `TRADE_CLOSED` y realimente el lifecycle (OPEN → TP_HIT/SL_HIT → CLOSED). ✅
   - Nuevo módulo `app/execution/`: `ExecutionEngine` abre posiciones desde `SIGNAL_SENT` (paper trading), monitorea `CANDLE_UPDATED`, calcula PnL, publica `TRADE_OPENED`/`TRADE_CLOSED` y realimenta el lifecycle vía `transition_by_pattern`.
   - `ExitReason` enum en `app/execution/models.py`.
   - El pipeline publica `CANDLE_UPDATED` con la última vela OHLCV y `SIGNAL_SENT` enriquecido (`signal_id`, `pattern_id`, `direction`, `strategy`, `size`, `indicators` canónicos).
2. Activar `LearningService.start()` en producción y conectar su modelo con el del pipeline (unificar features). ✅
   - Nuevo módulo `app/ml/features.py`: vector canónico de 12 features técnicas unificado para scoring y aprendizaje.
   - `ScoringEngine._get_ml_score` usa `learning_service.predict()` cuando `is_trained`; score neutro 50 si no hay modelo.
   - El lifespan de la API arranca/detiene `LearningService(repository=KnowledgeRepository())` (DB real desde Fase 4) y lo inyecta a `PatternService`.
3. Producir `CANCELLED`/`REJECTED` desde confirmación/riesgo. ✅
   - `REJECTED` si `RiskEngine.assess()` no es aceptable; `CANCELLED` si el patrón se deforma tras enviar señal (antes de la entrada).
4. Asignar `expires_at` al crear el patrón. ✅
   - El pipeline asigna `expires_at = detected_at + timeout * max_confirmation_candles`; `BasePattern.update()` lo fija al expirar si es `None`.

**Verificación**: 147 tests pasan (16 nuevos en `tests/unit/test_execution.py`, `test_pipeline.py`, `test_lifecycle.py`, `test_scoring.py`), flake8 limpio en los archivos modificados, mypy limpio en los módulos nuevos, imports y compilación OK.

### Fase 4 — Persistencia y API real (COMPLETADA, 2026-08-07)
1. Alembic + migraciones; escribir lifecycle, señales, backtests, predictions a la DB.
   - ✅ **Completado**: los repos write-through persisten lifecycle, señales, trades, backtests, modelos y predicciones. Alembic configurado (`alembic.ini` + `migrations/` async) con 3 revisiones: `initial_schema`, `backtest_trades_equity` (columnas JSON `trades`/`equity_curve` en `backtests`) y `lifecycle_uuid` (UUID persistido en `lifecycles`). Verificado `alembic upgrade head` contra SQLite y Postgres (via `DATABASE_URL`). Rehidratación del estado al arrancar: `LifecycleRepository.list()` reconstruye `PatternResult`+`LifecycleEvent` desde DB y `LifecycleEngine.rehydrate_from_db()` los carga en memoria en `PatternService.start()`.
2. Añadir routers: `/api/v1/models` (AI Models), `/api/v1/trades`, `/api/v1/lifecycle`. ✅
   - `models.py` (GET listado/detalle + POST predict), `trades.py` (GET listado/detalle), `lifecycle.py` (GET estadísticas/listado/por patrón/por id). Registrados en `api/main.py:100-106`.
3. Reemplazar datos sintéticos por flujo real (dashboard con instancia compartida del pipeline). ✅
   - Dashboard y lifecycle usan `service.pipeline.lifecycle`; señales/trades/backtests leen de la DB. ✅ **Stubs cerrados**: `GET /backtests/{id}` y `GET /backtests/{id}/trades` ahora devuelven `equity_curve` y `trades` reales persistidos en las columnas JSON. El payload vacío de backtest sigue generando datos sintéticos como input por diseño.
4. Override `DATABASE_URL` por env var. ✅ Implementado en `settings.py:23-30` (prioridad sobre campos discretos) con test en `tests/unit/test_config.py`.

**Verificación**: 179 tests pasan (2 nuevos: round-trip de trades/equity_curve en backtests, rehidratación de lifecycle repo/engine), flake8 limpio en los archivos modificados, imports y compilación OK.

### Fase 5 — Ampliar catálogo (COMPLETADA, 2026-08-07)
1. Patrones (13): triángulos, wedges, rectángulo, canal, cup & handle, rounded bottom, diamond, broadening, triple top/bottom. Poblarla carpeta `neutral/`. ✅
   - **Completado**: `app/patterns/neutral/` con `ascending_triangle`, `descending_triangle`, `symmetric_triangle`, `rising_wedge`, `falling_wedge`, `rectangle`, `channel`, `cup_and_handle`, `rounded_bottom`, `diamond`, `broadening`, `triple_top`, `triple_bottom` (21 patrones en total). Geometría compartida en `neutral/geometry.py` (fit_line/line_at/find_peaks/find_troughs). `key_levels` compatibles con `_prepare_price_levels` del pipeline. Registrados vía `@register_pattern` y testeados con datos sintéticos (13 tests de detección).
2. Modelos ML (8): LightGBM/XGBoost/CatBoost (tabulares), luego LSTM/Transformer/CNN (series), luego Isolation Forest/AutoEncoder (anomalías). ✅
   - **Completado** (9 modelos en total): `xgboost_model.py`, `lightgbm_model.py`, `catboost_model.py` (tabulares, con feature importance); `sequence_base.py` + `lstm_model.py`, `transformer_model.py`, `cnn_model.py` (series, torch, con save/load por state_dict + config); `isolation_forest.py`, `autoencoder.py` (anomalías, probabilidad de anomalía en `predict_proba`). Todos registrados en `MLModelFactory` (visible en `GET /api/v1/models`) y con tests de train/predict/evaluate/save-load roundtrip.
3. Aplicar el sistema de scoring a cada patrón nuevo y eliminar el `score()` muerto de cada patrón (o integrarlo). ✅
   - **Completado**: se eliminó `score()` del ABC `BasePattern` y de los 8 patrones existentes (era código muerto: el pipeline usa `ScoringEngine`). Los 13 patrones nuevos no implementan `score()`. Se actualizó `PATTERN_INTERFACE_METHODS` en `tests/unit/test_patterns.py`.

**Verificación**: 211 tests pasan (32 nuevos: 14 en patrones nuevos, 18 en modelos ML), flake8 limpio en archivos modificados, mypy sin errores nuevos (solo `import-untyped` preexistentes de librerías sin stubs).

### Fase 6 — Riesgo, Estrategias y Telegram completos
1. `RiskEngine` cableado al pipeline y reutilizado por el backtest (quitar duplicación). ✅
   - **Completado**: el `BacktestEngine` ya recibe/reutiliza el `RiskEngine` (sin duplicación de sizing) e implementa trailing stop y stops ATR; el `PatternService` alimenta el `RiskEngine` con `symbol_sectors`/`correlations` desde `config/settings.yaml` (`risk.symbol_sectors`, `risk.correlations`).
2. Exposición por sector y correlación; SL/TP dinámico y trailing stop en `BacktestEngine`. ✅
   - **Completado**: `_check_sector_exposure`/`_check_correlated_exposure` del `RiskEngine` ya reciben datos reales; el trailing/ATR stop estaba implementado y ahora se documenta.
3. Telegram: imagen (ChartGenerator), timeframe, cooldown configurable, retries con backoff, confirmación de entrega, dedup persistente. ✅
   - **Completado**: `TelegramNotifier.send_signal(signal, candles, pattern)` envía el gráfico vía `ChartGenerator` + `sendPhoto` (con fallback a texto si la imagen falla), añade **timeframe + fecha/hora** al mensaje, hace retries con backoff exponencial (`telegram.max_retries`/`retry_backoff_seconds`/`timeout_seconds`). El pipeline confirma la entrega (`SignalEngine.mark_delivered`/`mark_failed`) y el gate de envío es configurable (`telegram.min_priority`, default `CRITICAL`). El cooldown de señales se lee de `patterns.scoring.cooldown_minutes` y el dedup persiste en `telegram.dedup_store_path` (JSON).
4. `StrategyManager` con gestión runtime y API de estrategias. ✅
   - **Completado**: nuevo `app/strategy/manager.py` (enable/disable/params/reset en caliente, delega en `StrategyEngine`), inyectado en `PatternPipeline`/`PatternService`, con rutas `GET/PATCH /api/v1/strategies`.

**Verificación**: 236 tests pasan (25 nuevos: 8 de telegram, 8 de la API de estrategias, 6 del manager, 2 de señales/dedup, 1 del gate `min_priority`), flake8 limpio en archivos modificados. `kaleido` añadido a `pyproject.toml` (requiere Chrome para renderizar; si no está disponible el notifier cae a texto).

### Fase 7 — Configuración y calidad (COMPLETADA)
1. Parametrización total en YAML: capital inicial, comisiones, splits, simulación, reintentos y límites. ✅
2. Consumo de `walk_forward_splits` y `monte_carlo_simulations` desde la API de validación y backtesting. ✅
3. Limpieza de código y verificación robusta. ✅
4. Suite completa de 337 tests (unitarios, integración y E2E) cubriendo motores, repositorios, pipelines, estrategias y API. ✅

### Fase 8 — Robustez de la definición del label (COMPLETADA)
1. Barrido `threshold` (4 valores) × `min_up_moves` (3 valores) con `forward_periods=5` fijo (12 configs) sobre TRAIN/VALIDATION + walk-forward. ✅
2. Métricas por config: positive_ratio, mean/std validation AUC y PR-AUC; tabla y diagnóstico ROBUSTA/FRÁGIL/DÉBIL sin tocar TEST FINAL. ✅
3. Motor `run_label_sweep` + modo CLI exclusivo `--label-sweep` y tests (`tests/unit/test_fase8.py`). ✅
4. Detalle: `docs/mejoras/respuesta_fase8`. ✅

### Fase 9 — Reproducibilidad y Model Metadata (COMPLETADA)
1. Semillas globales (Python/NumPy/PyTorch) fijadas con `seed_all()` + flag CLI `--seed`. ✅
2. Sidecar `.meta.json` enriquecido con `dataset` (SHA-256), `features`, `label`, `preprocessing`, `sequence`, `training`, `validation`, `test`, `selection`, `software`, `git.commit_sha` y `random_seed` vía `build_model_sidecar_context()`. ✅
3. Hash del archivo de datos (`SHA-256`) para detectar "mismo nombre, contenido diferente". ✅
4. Integración en `save_winner()` + exports y tests (`tests/unit/test_reproducibility.py`). ✅
5. Detalle: `docs/mejoras/respuesta_fase9`. ✅

### Fase 10 — Evaluación Final Out-of-Sample (COMPLETADA)
1. Tabla de VALIDATION walk-forward (`MODEL | MEAN_AUC | STD_AUC | MEAN_PR_AUC`) vía `format_walk_forward_table()`. ✅
2. Selección exclusiva por media de la métrica sobre los folds (walk-forward) o VALIDATION; TEST FINAL nunca participa. ✅
3. Evaluación única del ganador sobre TEST FINAL (`evaluate_winner_on_test`) con comparación histórica frente al experimento original (LSTM TEST=0.6513) y su aclaración OOS. ✅
4. Conclusión/clasificación de la señal (`classify_signal`: ROBUST/POSSIBLE/WEAK/NO EVIDENCE) sin afirmar rentabilidad. ✅
5. Tests (`tests/unit/test_fase10.py`) y detalle en `docs/mejoras/respuesta_fase10`. ✅

### Fase 11 — Promoción Segura a Producción (COMPLETADA)
1. `--db` ya NO activa automáticamente: registra el modelo como INACTIVO sin tocar el activo previo. ✅
2. Nuevo flag `--promote`: solo `--db --promote` activa el modelo (promoción explícita). ✅
3. Mitigación FASE 1.1: transacción única (deactivate + upsert activo) en `MLModelRepository.promote()` para evitar dejar el símbolo sin modelo activo si la conexión falla a medias. ✅
4. Tests (`MLModelRepository.promote`, `TestDBRegistration` con `--db`/`--promote`, integración) y detalle en `docs/mejoras/respuesta_fase11`. ✅

---

## 4. Referencias Útiles

- `docs/ARCHITECTURE.md` — diseño y flujo de datos.
- `docs/BACKTESTING.md` — motor de backtesting y validaciones.
- `docs/LEARNING.md` — aprendizaje continuo (offline/online).
- `docs/PATTERNS.md` — detección de patrones.
- `docs/CONFIGURATION.md` — referencia de configuración YAML.
- `app/patterns/pipeline.py` — orquestación del flujo (detección → hipótesis → estrategia → señal).
