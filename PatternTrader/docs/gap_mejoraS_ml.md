# Plan de Implementación: Entrenamiento Multi-Modelo, Comparación y Selección por Par (`gap_mejoraS_ml.md`)

## 1. Introducción y Objetivos

Actualmente, el script `train_model.py` entrena exclusivamente el modelo `RandomForestModel` utilizando un conjunto de datos OHLCV, separación de entrenamiento y pruebas (Train/Test split) y cálculo de indicadores técnicos. 

El objetivo de este plan es diseñar e implementar un sistema avanzado de entrenamiento (`train_and_compare.py`) que permita:
1. **Soportar los 9 modelos de ML** definidos en la plataforma (`random_forest`, `xgboost`, `lightgbm`, `catboost`, `lstm`, `transformer`, `cnn`, `isolation_forest`, `autoencoder`) a través de la factoría `MLModelFactory`.
2. **Separar robustamente datos de entrenamiento y test** (respetando orden cronológico, `shuffle=False`).
3. **Comparar resultados y métricas** (Accuracy, Precision, Recall, F1-Score, ROC-AUC) de todos los modelos entrenados en una misma corrida.
4. **Identificar automáticamente el mejor modelo** para un par y timeframe específicos según una métrica objetivo (ej. ROC-AUC o F1).
5. **Persistir el modelo ganador** por par (evitando sobrescribir modelos de otros activos, ej. `rf_usdcad.pkl` vs `rf_usdjpy.pkl`) y registrarlo en la base de datos con `MLModelRepository` (`is_active=True`).
6. **Integrar dinámicamente el mejor modelo** en el motor de evaluación y scoring (`ScoringEngine`) y en el pipeline de trading.

---

## 2. Arquitectura del Script de Entrenamiento y Comparación (`train_and_compare.py`)

Tomando como base `train_model.py`, el nuevo script unificado operará bajo la siguiente estructura:

```
PatternTrader/
├── train_and_compare.py        # Script centralizado multi-modelo
├── models/
│   ├── best_random_forest_usdcad.pkl
│   ├── best_xgboost_usdcad.json
│   └── ...
```

### 2.1 Flujo de Ejecución del Script
1. **Carga y Limpieza de Datos**: Lectura de archivo CSV/TXT (OHLCV tab-delimitado o coma).
2. **Ingeniería de Características (Features)**: Cálculo de indicadores técnicos (`rsi`, `macd`, `ema_21`, `ema_50`, `atr`, `volume_ratio`, `price_change`, `high_low_range`, `close_position`, `trend_strength`).
3. **Generación de Etiquetas (Labels)**: Cálculo de retornos futuros con base en `--forward-periods` y `--threshold`.
4. **Formateo de Datos**:
   - **Tabulares (2D)**: Para Random Forest, XGBoost, LightGBM, CatBoost, Isolation Forest.
   - **Secuenciales (3D)**: Reshape en ventanas temporales (ej. `sequence_length=30`) para LSTM, Transformer, CNN, AutoEncoder.
5. **Entrenamiento y Evaluación en Lote**:
   - Iterar sobre la lista de modelos especificada (o `--model all`).
   - Ejecutar `.train()` y `.evaluate()` en cada modelo.
   - Recolectar métricas en un DataFrame de resumen.
6. **Comparación y Selección**:
   - Ordenar por la métrica objetivo (ej. `roc-auc`).
   - Mostrar tabla comparativa formateada en consola.
   - Guardar el artefacto ganador en el directorio de modelos con sufijo de par (ej. `models/{model_name}_{symbol}.pkl`).
   - Registrar en base de datos mediante `MLModelRepository` marcando el modelo como activo para ese par.

---

## 3. Especificación Técnica de los Modelos Soportados

| Modelo | Tipo | Tipo de Entrada | Archivo Base |
|--------|------|-----------------|--------------|
| `random_forest` | Tabular | 2D `(N, Features)` | `app/ml/models/random_forest.py` |
| `xgboost` | Tabular | 2D `(N, Features)` | `app/ml/models/xgboost_model.py` |
| `lightgbm` | Tabular | 2D `(N, Features)` | `app/ml/models/lightgbm_model.py` |
| `catboost` | Tabular | 2D `(N, Features)` | `app/ml/models/catboost_model.py` |
| `lstm` | Secuencia | 3D `(N, SeqLen, FeatDim)` | `app/ml/models/lstm_model.py` |
| `transformer` | Secuencia | 3D `(N, SeqLen, FeatDim)` | `app/ml/models/transformer_model.py` |
| `cnn` | Secuencia | 3D `(N, SeqLen, FeatDim)` | `app/ml/models/cnn_model.py` |
| `isolation_forest` | Anormalidad | 2D `(N, Features)` | `app/ml/models/isolation_forest.py` |
| `autoencoder` | Anormalidad | 2D o 3D | `app/ml/models/autoencoder.py` |

---

## 4. Integración con el Motor de Scoring (`ScoringEngine`)

Para asegurar que la estrategia utilice por defecto el modelo óptimo entrenado para el par correspondiente:

1. **Modificación en `app/scoring/engine.py`**:
   - Actualizar `_load_ml_model(model_path, symbol=None)` para buscar primero el artefacto específico del par activo (ej. `*_{symbol}.*` o consultar `MLModelRepository` filtrando por activo activo).
   - Si no existe un modelo específico para el par, aplicar fallback al modelo general o estático.
2. **Uso en el Pipeline (`PatternPipeline`)**:
   - Al procesar un símbolo (ej. `BTCUSDT` o `USDCAD`), el `ScoringEngine` inyecta automáticamente el modelo correspondiente en el componente `ml_history` del sistema de puntuación.

---

## 5. Plan de Fases de Implementación

### Fase 1: Desarrollo de `train_and_compare.py`
- [x] Crear la estructura base de argumentos CLI (`--data-file`, `--symbol`, `--model`, `--metric`, `--save-dir`).
- [x] Integrar el preprocesamiento de datos y feature engineering compartido.
- [x] Implementar el bucle de instanciación y entrenamiento para modelos tabulares y secuenciales mediante `MLModelFactory`.

### Fase 2: Comparación y Persistencia por Par
- [x] Implementar la evaluación unificada (Accuracy, Precision, Recall, F1, ROC-AUC).
- [x] Generar salida tabular en consola comparando todos los modelos.
- [x] Implementar guardado de artefactos con nomenclatura por par (`{model_name}_{symbol}`).
- [x] Conectar con `MLModelRepository` para persistencia en base de datos.

### Fase 3: Integración en el Pipeline y Scoring
- [x] Actualizar `ScoringEngine` para carga dinámica basada en el símbolo del activo.
- [x] Añadir pruebas unitarias y de integración para el nuevo script de entrenamiento comparativo.

#### Resumen de la Fase 3 (implementada)

1. **Rehidratación por símbolo en `ScoringEngine`** (`app/scoring/engine.py`):
   - `_load_ml_model_for_symbol(symbol)` busca los sidecar `*_{symbol}.meta.json` del par, rehidrata la clase correcta vía `MLModelFactory.create_new` y cachea la instancia.
   - **Corrección**: el patrón de búsqueda usa guion bajo (`*_{symbol}.meta.json`), acorde con la nomenclatura real de `save_winner` (antes usaba punto y nunca encontraba el artefacto).
   - **Candidatos múltiples**: si coexisten varios ganadores para el mismo símbolo (reentrenamientos con distintos modelos), se prioriza el sidecar con `trained_at` más reciente y, si su artefacto falla al cargar, se degrada al siguiente candidato.
   - `ScoringEngine` acepta ahora un `model_path` opcional para facilitar pruebas y despliegues con directorios alternativos.
   - El `PatternPipeline` no requiere cambios: el `ScoringEngine` inyecta automáticamente el modelo del par en el componente `ml_history` del scoring.

2. **Correcciones en `train_and_compare.py`**:
   - `--model` con `default=None` + `action="append"` (antes `default="all"` rompía argparse o entrenaba `all` junto al modelo pedido). Ahora también soporta valores separados por coma (`--model random_forest,xgboost`).
   - `derive_timeframe` ahora reconoce timeframes con prefijo numérico (`1h`, `4h`, `1d`) y no solo `H1`.

3. **Pruebas añadidas**:
   - `tests/unit/test_scoring.py::TestScoringPerSymbolModel`: carga del ganador por par, símbolo desconocido → `None`, selección del sidecar más reciente, degradación cuando el artefacto más nuevo falta, y uso efectivo del modelo del par en `calculate_score`.
   - `tests/unit/test_ml_training.py::TestTrainAndCompareCLI`: `derive_symbol`/`derive_timeframe` y ejecución completa de `main()` con `--no-save` y con persistencia del ganador.
   - `tests/integration/test_learning_integration.py::test_register_in_db_activates_per_symbol`: `register_in_db` registra el ganador como activo y desactiva el modelo previo del mismo símbolo.
