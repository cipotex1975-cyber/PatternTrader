# Análisis de Gaps y Estado del Proyecto

> **Fecha de la auditoría**: 2026-08-07 (revisada)
> **Alcance**: Revisión del código en `app/`, `tests/`, `config/` y scripts raíz contra los requisitos originales del proyecto (prompt de arquitectura) y la documentación en `docs/`.
> **Objetivo**: Documentar el estado actual, los gaps y las recomendaciones para poder retomar el desarrollo desde este punto de forma ordenada.

---

## Resumen Ejecutivo

PatternTrader tiene una **base sólida**: la arquitectura Clean/DDD está bien implementada, el backtesting es el módulo más maduro y completo, y hay **177 tests unitarios** que se recolectan sin errores. Desde la auditoría inicial se cerraron las fases 1-3 (bugs de runtime, arquitectura de estrategias, ciclo de vida completo) y la **fase 4 está muy avanzada**: los repositorios DB están implementados y cableados como *write-through* a los motores, y la API expone routers reales de trades, lifecycle, señales y modelos ML con persistencia. Los gaps principales restantes: **faltan 13 de 21 patrones** y **8 de 9 modelos ML**, no hay migraciones Alembic, y el cooldown/telegram siguen sin configuración completa.

Estado por módulo:

| Área | Estado | Nota |
|------|--------|------|
| Arquitectura | ✅ Completa | Clean/DDD/Hexagonal, Factory/Registry/Observer |
| Data Providers | ✅ Completo | 7 providers funcionales y testeados |
| Market Engine | ✅ Completo | Pivots, zigzag, fractales, trendlines, canales, indicadores |
| Patrones | ⚠️ 8/21 | Faltan 13; carpeta `neutral/` vacía |
| Lifecycle | ✅ Fase 3+4 | Ciclo completo realimentado por trades; persistencia write-through (`lifecycles`+`patterns`) |
| Health | ✅ Completo | 8 factores, pesos en YAML |
| Scoring | ✅ Completo | Pesos en YAML; usa modelo de conocimiento cuando está entrenado |
| Confirmación | ✅ Corregida | Ruptura direccional resuelta (§1.1); spread sigue placeholder |
| Señales | ✅ Fase 4 | Persistencia DB (`signals`); dedup/cooldown ok; cooldown hardcodeado |
| ML | ⚠️ 1/9 | Solo Random Forest; API de modelos + persistencia de predicciones |
| Aprendizaje continuo | ✅ Conectado | Recibe `TRADE_CLOSED`/`TRADE_OPENED`; registra el modelo entrenado en `ml_models` |
| Backtesting | ✅ Completo | Motor + validaciones + optimización + métricas; resultados persistidos en DB |
| Riesgo | ⚠️ Parcial | Cableado al pipeline (REJECTED); faltan sector/correlación/trailing |
| Estrategia | ✅ Fase 2 completa | Registry/Factory + 3 estrategias cableadas al pipeline |
| Telegram | ⚠️ Parcial | Sin imagen, retries, confirmación, timeframe |
| DB | ✅ Completa | 12 tablas + `knowledge_entries`; repos write-through funcionales; sin Alembic |
| API | ✅ Fase 4 | Routers de models/trades/lifecycle/signals reales y con persistencia |
| Config | ⚠️ Parcial | Varios valores hardcodeados pese a existir en YAML |
| Tests | ✅ 177 unitarios | Engines, DB, API, backtesting, learning, estrategias cubiertos |

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
- **Persistencia**: `LifecycleRepository` (write-through) persiste `patterns` + `lifecycles` en la misma sesión; está cableado al `PatternPipeline` vía `PatternService` (`patterns/service.py:39`) y a la ruta DB `asset` por símbolo.
- **Dashboard/lifecycle API**: `api/routes/lifecycle.py` y `dashboard.py` usan la **instancia compartida** `service.pipeline.lifecycle` (gap de instancias separadas resuelto).
- **Gaps restantes**:
  - Las transiciones leídas desde DB no se rehidratan en memoria (los repos son write-only para runtime; la API lee del dict en memoria, no de la DB).
  - `dashboard`/`lifecycle` muestran el estado del proceso actual; si la API se reinicia, el historial DB no se recarga.

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

### 2.11 Base de Datos (`app/database/`)

- Las **12 tablas requeridas existen** (+1 extra `knowledge_entries`): `assets`, `candles`, `indicators`, `patterns`, `lifecycles`, `signals`, `trades`, `backtests`, `predictions`, `ml_models`, `metrics`, `logs`. ✅
- PostgreSQL configurable vía campos discretos → `postgresql+asyncpg://` (`settings.py:13-27`, `settings.yaml:13-21`).
- **Fase 4 — repositorios write-through implementados y cableados** (`app/database/repositories/`):
  - `LifecycleRepository` → `LifecycleEngine` (registra `patterns`+`lifecycles` y actualiza transiciones).
  - `SignalRepository` → `SignalEngine` (`add`, `update_status`).
  - `TradeRepository` → `ExecutionEngine` (`add`, `update_closed`).
  - `MLModelRepository` → `LearningService.train_offline` (upsert).
  - `PredictionRepository` → `POST /api/v1/models/{name}/predict`.
  - `BacktestRepository` → rutas de backtests (add/get/list).
  - `AssetRepository` → resuelve la FK `assets.id` requerida por `patterns`.
  - `MetricRepository`/`LogRepository` → escritura de métricas/logs.
- **Bug corregido**: `get_async_session()` era un async generator usado con `async with` → TypeError en runtime (ningún repo llegaba a ejecutarse). Se envolvió con `@asynccontextmanager` (`database/base.py:69`); todos los repos funcionan y están testeados contra SQLite en memoria.
- **Gaps**:
  - No hay override de `DATABASE_URL` por variable de entorno.
  - No hay **Alembic/migraciones** pese a que `alembic` es dependencia declarada (carpeta `migrations/` existe pero no está conectada a un flujo de revisiones).
  - Los repos son de write-through: la API lee del estado en memoria de los motores, no rehidrata desde DB al arrancar.

### 2.12 API REST (`app/api/`)

**Routers presentes** (`api/main.py:98-106`): health, patterns, signals, trades, backtests, learning, dashboard, lifecycle, models.

| Grupo requerido | Estado |
|-----------------|--------|
| Patterns | ✅ |
| Dashboard | ✅ Backed por la instancia compartida del pipeline + repos de trades |
| Backtests | ✅ 10 endpoints; resultados persistidos (`BacktestRepository`); datos sintéticos solo si el payload viene vacío |
| Training | ⚠️ Solo `POST /learning/train` |
| Statistics | ⚠️ Solo por patrón |
| AI Models | ✅ `GET /models`, `GET /models/{name}`, `POST /models/{name}/predict` (persiste predicciones) |
| Signals | ✅ `GET /signals` y `GET /signals/{id}` leen de la DB (SignalRepository) |
| Trades | ✅ `GET /trades` y `GET /trades/{id}` leen de la DB (TradeRepository) |
| Health | ✅ |
| Lifecycle | ✅ `GET /lifecycle` (listado, estadísticas, por patrón, por id) sobre la instancia del pipeline |

**Rutas stub restantes**: en backtests, `equity_curve` y la lista de `trades` se devuelven **vacías** (`backtests.py:141, 162`) aunque `BacktestResult` las guarda; el payload vacío genera candles/patrones sintéticos (`_generate_candles`/`_generate_patterns`). No hay endpoints de escritura para señales/trades/lifecycle (solo lectura).

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

- **177 tests** en `tests/unit/` (recolectan y pasan); `tests/integration/` y `tests/e2e/` solo con `__init__.py` (**cero tests**).
- **Fase 4 — cobertura nueva** (`conftest.py` con fixture `sync_db`, SQLite en memoria con `StaticPool`):
  - `tests/unit/test_repositories.py` (11): round-trip de los 9 repos (`asset`, `trade`, `signal`, `lifecycle`, `backtest`, `mlmodel`, `prediction`, `log`, `metric`) contra la DB.
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

### Fase 4 — Persistencia y API real (EN CURSO, 2026-08-07)
1. Alembic + migraciones; escribir lifecycle, señales, backtests, predictions a la DB.
   - ⚠️ **Parcial**: los repos write-through ya persisten lifecycle, señales, trades, backtests, modelos y predicciones. Falta **Alembic** y rehidratar el estado desde DB al arrancar.
2. Añadir routers: `/api/v1/models` (AI Models), `/api/v1/trades`, `/api/v1/lifecycle`. ✅
   - `models.py` (GET listado/detalle + POST predict), `trades.py` (GET listado/detalle), `lifecycle.py` (GET estadísticas/listado/por patrón/por id). Registrados en `api/main.py:100-106`.
3. Reemplazar datos sintéticos por flujo real (dashboard con instancia compartida del pipeline). ✅
   - Dashboard y lifecycle usan `service.pipeline.lifecycle`; señales/trades/backtests leen de la DB. Quedan stubs: `equity_curve` y `trades` vacíos en backtests; payload vacío de backtest sigue generando datos sintéticos.
4. Override `DATABASE_URL` por env var. ❌ Pendiente.

**Verificación**: 177 tests pasan (30 nuevos: repos, SignalEngine, API de models; + `conftest.py` con `sync_db`), flake8 limpio en los archivos modificados, mypy limpio en motores/repos/rutas nuevos, imports y compilación OK.

### Fase 5 — Ampliar catálogo
1. Patrones (13): triángulos, wedges, rectángulo, canal, cup & handle, rounded bottom, diamond, broadening, triple top/bottom. Poblarla carpeta `neutral/`.
2. Modelos ML (8): LightGBM/XGBoost/CatBoost (tabulares), luego LSTM/Transformer/CNN (series), luego Isolation Forest/AutoEncoder (anomalías).
3. Aplicar el sistema de scoring a cada patrón nuevo y eliminar el `score()` muerto de cada patrón (o integrarlo).

### Fase 6 — Riesgo y Telegram completos
1. `RiskEngine` cableado al pipeline y reutilizado por el backtest (quitar duplicación).
2. Exposición por sector y correlación; SL/TP dinámico y trailing stop en `BacktestEngine`.
3. Telegram: imagen (ChartGenerator), timeframe, cooldown configurable, retries con backoff, confirmación de entrega, dedup persistente.

### Fase 7 — Configuración y calidad
1. Mover hardcodeos a YAML (cooldown, fees, capital inicial, tolerancias, `max_patterns_per_symbol`, `recalculate_interval_seconds`).
2. Consumir `backtesting.walk_forward_splits`/`monte_carlo_simulations` desde la API.
3. Eliminar/limpiar código muerto (`optimizer/engine.py`, imports sin uso, `score()` de patrones, `Signal.is_expired` sin aplicar).
4. Tests de integración y e2e; tests de `BacktestRunner`, resto de rutas API y `run_backtest.py`. (Los repos DB, el `SignalEngine` y la ruta de models ya están cubiertos.)

---

## 4. Referencias Útiles

- `docs/ARCHITECTURE.md` — diseño y flujo de datos.
- `docs/BACKTESTING.md` — motor de backtesting y validaciones.
- `docs/LEARNING.md` — aprendizaje continuo (offline/online).
- `docs/PATTERNS.md` — detección de patrones.
- `docs/CONFIGURATION.md` — referencia de configuración YAML.
- `app/patterns/pipeline.py` — orquestación del flujo (detección → hipótesis → estrategia → señal).
