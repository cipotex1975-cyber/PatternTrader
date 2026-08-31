# Entrenamiento de Modelos ML

## Visión General

PatternTrader permite entrenar modelos de Machine Learning con datos históricos OHLCV para predecir el éxito de patrones chartistas.

---

## Script de Entrenamiento

**Archivo**: `train_model.py`

### Uso Básico

```bash
python train_model.py <archivo_datos>
```

### Ejemplo

```bash
python train_model.py app/datos_test/USDCAD_H1_201005311000_202606010000.txt
```

### Parámetros

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `data_file` | (requerido) | Ruta al archivo de datos OHLCV |
| `--test-size` | 0.2 | Proporción de datos para pruebas (20%) |
| `--n-estimators` | 100 | Número de árboles del Random Forest |
| `--max-depth` | 10 | Profundidad máxima de los árboles |
| `--forward-periods` | 5 | Velas hacia adelante para generar labels |
| `--threshold` | 0.001 | Umbral mínimo de cambio para label positivo (0.1%) |
| `--save-model` | None | Ruta para guardar el modelo entrenado |

### Ejemplo Completo

```bash
python train_model.py app/datos_test/USDCAD_H1_201005311000_202606010000.txt \
  --test-size 0.2 \
  --n-estimators 200 \
  --max-depth 15 \
  --forward-periods 10 \
  --threshold 0.002 \
  --save-model models/rf_usdcad.pkl
```

---

## Formato de Datos

El script acepta archivos tab-delimitados (formato MT4/MT5):

```
DateTime	time	Open	High	Low	Close	Tickvol	Volume	Spread
2010-05-31	10:00:00	1.04944	1.05073	1.04826	1.04919	1086	0	26
2010-05-31	11:00:00	1.0492	1.05057	1.04913	1.04968	759	0	20
```

### Columnas Requeridas

| Columna | Descripción |
|---------|-------------|
| `DateTime` | Fecha |
| `time` | Hora |
| `Open` | Precio de apertura |
| `High` | Precio máximo |
| `Low` | Precio mínimo |
| `Close` | Precio de cierre |
| `Tickvol` | Volumen de ticks |

---

## Features Calculados

El script genera automáticamente los siguientes indicadores técnicos:

| Feature | Indicador | Descripción |
|---------|-----------|-------------|
| `rsi` | RSI (14) | Índice de Fuerza Relativa |
| `macd` | MACD (12,26,9) | Convergencia/Divergencia de Medias Móviles |
| `macd_signal` | Señal MACD | Línea de señal del MACD |
| `macd_hist` | Histograma MACD | Diferencia MACD - Señal |
| `ema_21` | EMA 21 | Media Móvil Exponencial 21 períodos |
| `ema_50` | EMA 50 | Media Móvil Exponencial 50 períodos |
| `atr` | ATR (14) | Rango Verdadero Medio |
| `volume_ratio` | Ratio de Volumen | Volumen actual / Volumen promedio 20 períodos |
| `price_change` | Cambio de Precio | Variación porcentual |
| `high_low_range` | Rango H-L | (High - Low) / Close |
| `close_position` | Posición del Cierre | (Close - Low) / (High - Low) |
| `trend_strength` | Fuerza de Tendencia | (EMA21 - EMA50) / EMA50 |

---

## Labels (Objetivo)

El modelo predice si el precio subirá más del `--threshold` en `--forward-periods` velas:

- **1 (Positivo)**: Precio sube > threshold en N velas
- **0 (Negativo)**: Precio no sube suficiente

---

## Integración con la Aplicación

### Guardar Modelo

```bash
python train_model.py app/datos_test/USDCAD_H1_201005311000_202606010000.txt \
  --save-model models/rf_usdcad.pkl
```

El modelo se guarda en `models/` (configurado en `config/settings.yaml` → `ml.model_path`).

### Cargar Modelo en la Aplicación

```python
from app.ml.models.random_forest import RandomForestModel

# Crear instancia
model = RandomForestModel()

# Cargar modelo entrenado
model.load("models/rf_usdcad.pkl")

# Usar para predecir
import numpy as np
features = np.array([...])  # Features calculados
prediction = model.predict(features)
probability = model.predict_proba(features)
```

### Uso en ScoringEngine

El `ScoringEngine` en `app/scoring/engine.py` carga automáticamente el modelo ML **específico del símbolo** que se está evaluando:

1. **Carga por par** (`_load_ml_model_for_symbol`): busca el sidecar `*_{symbol}.meta.json` en `ml.model_path`, rehidrata la clase del artefacto con `MLModelFactory.create_new` y cachea la instancia. Si existen varios candidatos para el símbolo, prioriza el más reciente por `trained_at`.
2. **Fallback genérico**: si no hay modelo del par, usa el primer artefacto `*.pkl` sin sidecar por par (modelo estático).
3. **Aprendizaje continuo**: si `attach_knowledge` conectó un `LearningService` entrenado, ese modelo tiene prioridad sobre el por-par.

```python
from app.scoring.engine import ScoringEngine

# Directorio por defecto (config/settings.yaml → ml.model_path)
engine = ScoringEngine()

# Directorio alternativo (útil para testing/despliegue)
engine = ScoringEngine(model_path="/ruta/a/models/")
```

---

## Estructura de Archivos

```
PatternTrader/
├── train_model.py              # Script de entrenamiento
├── models/
│   └── rf_usdcad.pkl          # Modelo entrenado
├── config/
│   └── settings.yaml          # Configuración ml.model_path
├── app/
│   └── ml/
│       ├── base.py            # BaseMLModel
│       ├── factory.py         # MLModelFactory
│       └── models/
│           └── random_forest.py
└── app/datos_test/
    └── USDCAD_H1_*.txt        # Datos de entrenamiento
```

---

## Métricas de Evaluación

El script muestra:

| Métrica | Descripción |
|---------|-------------|
| Train accuracy | Precisión en datos de entrenamiento |
| Test accuracy | Precisión en datos de prueba |
| Precision | Verdaderos positivos / Predicciones positivas |
| Recall | Verdaderos positivos / Positivos reales |
| F1 | Media armónica de Precision y Recall |
| Feature importance | Importancia de cada feature |

---

## Mejores Prácticas

1. **Datos suficientes**: Mínimo 1000 muestras para entrenar
2. **Orden temporal**: No barajar datos (`shuffle=False`) para respetar cronología
3. **Validación**: Usar datos diferentes a los de entrenamiento
4. **Guardado**: Siempre guardar el modelo con `--save-model`
5. **Reentrenamiento**: Actualizar el modelo con nuevos datos periódicamente

---

## Ejemplo de Uso Completo

```bash
# 1. Entrenar modelo
python train_model.py app/datos_test/USDCAD_H1_201005311000_202606010000.txt \
  --save-model models/rf_usdcad.pkl

# 2. Verificar que el modelo se guardó
ls -la models/

# 3. Usar en la aplicación
python -c "
from app.ml.models.random_forest import RandomForestModel
model = RandomForestModel()
model.load('models/rf_usdcad.pkl')
print(f'Modelo cargado: {model.name}')
print(f'Entrenado: {model.is_trained}')
"
```

---

## Entrenamiento Avanzado Multi-Modelo y Comparación por Par (`train_and_compare.py`)

Implementado a partir del plan de mejoras (`docs/gap_mejoraS_ml.md`), este script permite entrenar y comparar **los 9 modelos de ML** de la plataforma utilizando datos históricos OHLCV (en formato CSV/TXT tab-delimitado).

### Características Principales
1. **Soporte Multi-Modelo**: Evalúa en una misma corrida modelos tabulares (`random_forest`, `xgboost`, `lightgbm`, `catboost`), secuenciales (`lstm`, `transformer`, `cnn`) y de anomalías (`isolation_forest`, `autoencoder`).
2. **Separación Cronológica**: Divide los datos respetando el orden temporal sin mezclar (`shuffle=False`).
3. **Métricas Profesionales**: Compara Accuracy, Precision, Recall, F1-Score, ROC-AUC y PR-AUC.
4. **Selección y Persistencia por Par**: Identifica automáticamente el mejor modelo según la métrica objetivo (ej. `--metric roc_auc`) y guarda el artefacto ganador en `models/` con sufijo de par (`{model_name}_{symbol}.{ext}`) junto a un archivo sidecar de metadatos (`.meta.json`).
5. **Integración con DB y Scoring**: Con la bandera `--db`, registra y activa el modelo ganador en la base de datos (`ml_models`). El `ScoringEngine` utiliza automáticamente este sidecar para rehidratar el modelo específico del activo en tiempo de ejecución.

### Ejemplo de Uso

```bash
python train_and_compare.py app/datos_test/USDCAD_H1_201005311000_202606010000.txt \
  --model all \
  --metric roc_auc \
  --forward-periods 5 \
  --threshold 0.001 \
  --db
```

### Parámetros Principales

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `data_file` | (requerido) | Ruta al archivo de datos OHLCV (tab-delimited) |
| `--model` | `all` | Modelo(s) a entrenar o `all` |
| `--metric` | `roc_auc` | Métrica objetivo (`accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`) |
| `--forward-periods` | 5 | Velas hacia adelante para etiquetado |
| `--threshold` | 0.001 | Retorno mínimo para label positivo |
| `--sequence-length` | 30 | Longitud de ventana temporal para modelos secuenciales |
| `--epochs` | 10 | Épocas de entrenamiento para modelos de PyTorch |
| `--feature-scaling` | `none` | Preprocessing de features: `none` (sin escalar) o `standard` (StandardScaler fit SOLO con TRAIN) |
| `--walk-forward-splits` | `1` | Validación walk-forward (Fase 5). Default `1` = desactivado (selección por VALIDATION de Fase 2). Con `N>1` selecciona por la media de la métrica sobre N folds expanding; el TEST FINAL queda aislado |
| `--db` | `False` | Registrar el ganador en la base de datos y marcarlo como activo |
| `--no-save` | `False` | No persistir el artefacto ganador en disco |

### Ejecución de la Fase 3 (Pipeline, Scoring y Pruebas)

La Fase 3 del plan (`docs/gap_mejoraS_ml.md`) está implementada:

- **Integración con el pipeline**: `PatternPipeline` usa `ScoringEngine`, que inyecta automáticamente el modelo entrenado del símbolo en el componente `ml_history` sin cambios adicionales.
- **Selección del mejor candidato**: si hay varios ganadores para un símbolo (reentrenamientos), el `ScoringEngine` elige el sidecar más reciente (`trained_at`) y degrada al siguiente si el artefacto falla.
- **Pruebas**:
  - Unitarias: `tests/unit/test_scoring.py` (rehidratación por símbolo) y `tests/unit/test_ml_training.py` (CLI completo de `train_and_compare.py`).
  - Integración (requiere Postgres): `tests/integration/test_learning_integration.py` → `register_in_db` activa el ganador del par y desactiva el anterior.

### Preprocessing / Feature Scaling (Fase 4)

La infraestructura de preprocessing reproducible está integrada en `train_and_compare.py` y en el `ScoringEngine`:

- **Flag** `--feature-scaling {none|standard}` (default `none`). Con `standard` se aplica un `StandardScaler` cuyo `fit` se hace **exclusivamente con TRAIN**; VALIDATION y TEST usan únicamente `scaler.transform()` (sin leakage). Con `none` las matrices pasan intactas (default de producción, respeta la conclusión de la Fase 3.2).
- **Orden documentado**: `raw features → scaler.transform → build sequences → model` (el scaling se aplica antes de construir las ventanas).
- **Modelos a los que aplica**: sensibles (LSTM, CNN, Transformer, AutoEncoder). No se fuerza en Random Forest, XGBoost, LightGBM ni CatBoost.
- **Persistencia**: al entrenar con `standard`, el ganador guarda un artefacto `{modelo}_{symbol}.scaler.json` y el sidecar `*.meta.json` registra un bloque `preprocessing`:

  ```json
  "preprocessing": {
    "type": "StandardScaler",
    "features": ["rsi", ... , "trend_strength"],
    "fitted_on": "TRAIN_ONLY"
  }
  ```

- **Serving**: el `ScoringEngine` rehidrata el scaler desde el sidecar al cargar el modelo de un par y lo reaplica en inferencia (`raw → scaler → sequence → model`), garantizando paridad entrenamiento/serving.
- **Tests**: `tests/unit/test_feature_scaling.py` verifica no-leakage (fit solo TRAIN), orden de features, round-trip sidecar→model y que servir equivale a predecir sobre datos ya escalados.

La activación es **opt-in**: el default sigue siendo `none` para no alterar las métricas actuales. Detalles completos en `docs/mejoras/respuesta_fase4`.

### Walk-Forward Validation (Fase 5)

La validación walk-forward mide la **estabilidad** del modelo a lo largo de
distintos períodos históricos usando una ventana expanding (sin shuffle, sin
futuro, sin overlap). Es **opt-in** mediante `--walk-forward-splits N>1`:

- Operación dentro del **conjunto de selección** (`TRAIN + VALIDATION` en orden
  cronológico); el **TEST FINAL queda aislado** y se evalúa una sola vez al final.
- Folds contiguos; cada validation fold es estrictamente posterior a su train.
- Anti-leakage de labels: cada `train_fold` y `validation_fold` recorta sus
  últimas `forward_periods` muestras (patrón OPCIÓN B de Fase 2). Verificado por
  `validate_walk_forward_no_future`.
- **Escaler por fold**: `fit` solo con el train del fold; validation solo
  `transform()`.
- **Secuenciales**: se reentrenan en cada fold con su propio context-build.
- **Selección**: el ganador se elige por la **media de la métrica objetivo sobre
  los folds** (`select_walk_forward_winner`); el TEST FINAL no participa.

Cada fold reentrena el modelo, lo cual es costoso (sobre todo para LSTM/CNN/
Transformer); por eso el default es OFF y se recomienda probar con `--model lstm`
y pocos splits. Detalles completos en `docs/mejoras/respuesta_fase5`.

