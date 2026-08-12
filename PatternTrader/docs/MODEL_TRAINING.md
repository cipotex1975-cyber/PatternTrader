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

El `ScoringEngine` en `app/scoring/engine.py` usa el modelo ML para evaluar patrones:

```python
from app.ml.models.random_forest import RandomForestModel
from app.core.config.settings import get_settings
from pathlib import Path

class ScoringEngine:
    def __init__(self) -> None:
        settings = get_settings()
        self._weights = settings.scoring.weights
        
        # Cargar modelo ML si existe
        model_path = Path(settings.ml.model_path) / "rf_usdcad.pkl"
        if model_path.exists():
            self._ml_model = RandomForestModel()
            self._ml_model.load(str(model_path))
        else:
            self._ml_model = None
    
    def _get_ml_score(self, features: np.ndarray) -> float:
        """Obtener score del modelo ML."""
        if self._ml_model is None or not self._ml_model.is_trained:
            return 50.0  # Default cuando no hay modelo
        
        try:
            proba = self._ml_model.predict_proba(features.reshape(1, -1))
            return float(proba[0][1]) * 100  # Probabilidad de éxito * 100
        except Exception:
            return 50.0
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
| `--db` | `False` | Registrar el ganador en la base de datos y marcarlo como activo |
| `--no-save` | `False` | No persistir el artefacto ganador en disco |

