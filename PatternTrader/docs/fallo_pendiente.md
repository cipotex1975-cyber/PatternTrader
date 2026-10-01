# Fallos pendientes

Estado verificado al cierre de la tarea "ML por símbolo + timeframe"
(ver `git log`/`git diff`; suite completa: **509 passed, 3 failed**).

Este documento recoge lo que **queda por arreglar**. Nada de lo aquí listado fue
introducido por el trabajo de ML por timeframe: son Tech Debt preexistente.

---

## 1. `test_sequence_models_train_predict_evaluate` falla (3 tests)

**Estado**: pendiente. Preexistente (ya fallaba antes de los cambios de ML).

```
tests/unit/test_ml_models.py::test_sequence_models_train_predict_evaluate[LSTMModel]
tests/unit/test_ml_models.py::test_sequence_models_train_predict_evaluate[TransformerModel]
tests/unit/test_ml_models.py::test_sequence_models_train_predict_evaluate[CNNModel]
```

### Causa raíz

Desajuste de contrato entre el test y lo que devuelve `SequenceBaseModel.train()`.

- `app/ml/models/sequence_base.py:273-285` devuelve:
  `epochs`, `train_loss`, `train_accuracy`, `validation_loss`,
  `validation_accuracy`, `best_epoch`, `best_validation_loss`,
  `early_stopping`, `patience`.
  **No incluye la clave `"loss"`.**
- `tests/unit/test_ml_models.py:126` afirma `assert "loss" in result`.
- Los modelos de árbol (`random_forest.py`, `xgboost_model.py`, …) sí devuelven
  `"loss"`, de ahí que solo fallen los tres de secuencia.

La clave `validation_loss: nan` que aparece en el fallo **no es un bug**: el test
llama a `model.train(X, y)` sin `X_val`/`y_val`, así que `last_val_loss` se queda
en `nan` por diseño.

### El código de producción ya lo tolera

`app/ml/training/compare.py:516` acepta ambos contratos:

```python
train_loss = train_metrics.get("loss", train_metrics.get("train_loss"))
```

Es decir, `train_and_compare.py` funciona con los nueve modelos. El único punto
que exige la clave `"loss"` es el test.

### Arreglo propuesto

Preferible: **arreglar el test**, no el modelo. Añadir `"loss"` a los modelos de
secuencia rompería la simetría con `best_validation_loss`/`early_stopping` y
obligaría a tocar la lógica de `compare.py`.

Opción mínima, en `tests/unit/test_ml_models.py:126`:

```python
assert "loss" in result or "train_loss" in result
```

Opción más estricta (recomendada, cubre el contrato real):

```python
assert "train_loss" in result
assert "train_accuracy" in result
assert "validation_loss" in result
```

### Al terminar

```bash
venv/bin/pytest tests/unit/test_ml_models.py -q
```

Debe quedar en 0 failed. No debería cambiar ningún otro test: nadie más lee
`"loss"` de modelos de secuencia.

---

## 2. `mypy`: 112 errores en 36 ficheros (preexistente)

**Estado**: pendiente, requiere triage. El recuento **no ha subido** con el
trabajo de ML por timeframe (112 antes y 112 después, pese a añadir
`app/ml/naming.py`). Por tanto no bloquea esta tarea, pero ensucia la
verificación.

Concentración de los errores:

| Fichero | Errores |
|---------|---------|
| `app/database/models.py` | 30 |
| `app/learning/repository.py` | 24 |
| `app/data/providers/alphavantage/provider.py` | 6 |
| `app/learning/service.py` | 5 |
| `app/data/providers/{polygon,metatrader,bybit,binance}/provider.py` | 4 c/u |
| resto (28 ficheros) | ≤ 3 c/u |

Sugerencia de orden: empezar por los dos primeros (54 de 112 errores) y
tipar las columnas de `app/database/models.py`, que arrastran al resto.

```bash
venv/bin/mypy app/
```

---

## 3. `test_pipeline_health_recalculation_is_throttled` — flaky (resuelto, documentado)

**Estado**: flaky por timing; **pasó** en la última suite completa. Falló en la
suite de referencia (antes de los cambios), luego pasó sin tocar nada.

Es un test sensible al reloj. Si vuelve a fallar de forma intermitente, revisar
el margen de tiempo del assertion; no mezclarlo con la tarea de ML.

---

## 4. `tests/e2e/test_full_flow.py` cuelga — YA DIAGNOSTICADO, NO ES UN BUG

**Estado**: resuelto. Se documenta para que nadie lo re-diagnostique.

### Síntoma aparente

`tests/e2e/test_full_flow.py::test_full_flow_persist_and_restart` se quedaba
colgado sin límite (había que matarlo con `timeout`/`SIGTERM`).

### Diagnóstico

Se descompone en dos partes, y **ninguna** es un defecto del test:

1. **Por qué estaba oculto al principio.** `requires_postgres`
   (`tests/conftest.py:32-35`) es un `skipif` que se evalúa **al importar**
   `conftest` y hace un probe de `TEST_DATABASE_URL`. La BD
   `pattern_trader_test` **no existía** en la primera ejecución, así que el
   probe fallaba y el test e2e se **skipeaba** en silencio. Durante esa misma
   corrida otro test con fixture `pg_db` la creó vía `_create_database()`
   (`tests/conftest.py:55-67`). A partir de ahí el probe pasó y el e2e empezó a
   ejecutarse de verdad.

2. **Por qué colgaba cuando sí se ejecutaba.** Mis propias ejecuciones previas
   habían sido terminadas con `SIGTERM` (`timeout ...` → `EXIT=143`). Al morir
   sin completar el teardown de `pg_db`
   (`tests/conftest.py:101`, `_truncate_all_tables()`), quedaron conexiones
   huérfanas de Postgres bloqueando. Con la BD truncada a mano y **cero backends
   residuales** en `pg_stat_activity`, el test pasó **3/3 intentos** en ~20 s.

   Descartado: no es estado sucio de la BD (truncar no lo arregló mientras había
   backends colgados) y no es código (cuelga igual en `main` limpio con
   `git stash`).

### Lección operativa

- **No mates la suite con `SIGTERM` y luego sigas usando la BD de test**: hay que
  cerrar las conexiones o la siguiente corrida se queda bloqueada.
- Si vuelve a aparecer un cuelgue en un test con `pg_db`, comprobar primero:
  ```sql
  SELECT pid, state, wait_event_type, wait_event, query
  FROM pg_stat_activity WHERE datname = 'pattern_trader_test';
  ```
  Si hay backends `idle in transaction` de procesos pytest muertos, ahí está la
  causa.
- Para un e2e se puede ejecutar con `-s -u` para ver salida en vivo.

---

## Resumen de verificación (suite completa)

```
509 passed, 3 failed  in 308.68s
```

| Comprobación | Estado |
|--------------|--------|
| `black app/ tests/ train_and_compare.py simulate_pipeline.py` | limpio |
| `isort app/ tests/ train_and_compare.py simulate_pipeline.py` | limpio |
| `flake8 app/ tests/` | limpio |
| `mypy app/` | 112 errores, **sin incremento** (ver §2) |
| `pytest -q` | 509 passed / 3 failed (§1) |
