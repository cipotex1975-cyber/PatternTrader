# Guía de Machine Learning

## Visión General

PatternTrader integra modelos de Machine Learning para predecir la probabilidad de éxito de los patrones chartistas detectados. El sistema está diseñado para aprender continuamente de cada operación.

---

## Arquitectura ML

```
┌─────────────────────────────────────────────────────────────┐
│                     ML Engine                               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ BaseMLModel │  │   Factory   │  │    Predictions      │ │
│  │ (Abstract)  │  │   Pattern   │  │     Storage         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Random    │  │  XGBoost    │  │       LSTM          │ │
│  │   Forest    │  │             │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Modelos Disponibles

### 1. Random Forest

**Archivo**: `app/ml/models/random_forest.py`

**Uso**:

```python
from app.ml.models.random_forest import RandomForestModel
import numpy as np

# Crear modelo
model = RandomForestModel(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# Datos de entrenamiento (ejemplo)
X_train = np.array([
    [0.85, 72, 150, 120, 250, 1.2],  # [confidence, rsi, macd, macd_signal, atr, volume_ratio]
    [0.78, 65, 100, 90, 200, 1.1],
    [0.92, 78, 180, 140, 300, 1.5],
    # ... más muestras
])

y_train = np.array([1, 0, 1, ...])  # 1 = éxito, 0 = fallo

# Entrenar
metrics = model.train(X_train, y_train, feature_names=[
    "confidence", "rsi", "macd", "macd_signal", "atr", "volume_ratio"
])

print(f"Accuracy: {metrics['train_accuracy']:.4f}")

# Predecir
X_test = np.array([[0.88, 70, 160, 130, 280, 1.3]])
prediction = model.predict(X_test)
probability = model.predict_proba(X_test)

print(f"Predicción: {prediction[0]}")
print(f"Probabilidad de éxito: {probability[0][1]:.2%}")

# Guardar modelo
model.save("models/random_forest_v1.pkl")

# Cargar modelo
model.load("models/random_forest_v1.pkl")
```

**Hiperparámetros**:

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| n_estimators | 100 | Número de árboles |
| max_depth | 10 | Profundidad máxima |
| random_state | 42 | Semilla aleatoria |

---

### 2. XGBoost (Próximamente)

**Uso**:

```python
from app.ml.models.xgboost import XGBoostModel

model = XGBoostModel(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)

model.train(X_train, y_train)
prediction = model.predict_proba(X_test)
```

---

### 3. LightGBM (Próximamente)

**Uso**:

```python
from app.ml.models.lightgbm import LightGBMModel

model = LightGBMModel(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)

model.train(X_train, y_train)
```

---

### 4. LSTM (Próximamente)

**Uso**:

```python
from app.ml.models.lstm import LSTMModel

model = LSTMModel(
    sequence_length=60,
    hidden_size=128,
    num_layers=2,
    dropout=0.2
)

# Para LSTM, los datos deben ser secuencias
X_train_seq = X_train.reshape(-1, 60, n_features)
model.train(X_train_seq, y_train)
```

---

## Interfaz BaseMLModel

Todos los modelos implementan esta interfaz:

```python
from app.ml.base import BaseMLModel, MLPrediction
import numpy as np

class BaseMLModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre del modelo."""
        ...
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Tipo: 'classification' o 'regression'."""
        ...
    
    @abstractmethod
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Entrenar el modelo. Retorna métricas."""
        ...
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predecir clases."""
        ...
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predecir probabilidades."""
        ...
    
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Evaluar modelo. Retorna accuracy, precision, recall, f1."""
        ...
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Guardar modelo a disco."""
        ...
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Cargar modelo desde disco."""
        ...
    
    def get_prediction(
        self,
        features: np.ndarray,
        symbol: str,
        timeframe: str,
        pattern_name: str,
    ) -> MLPrediction:
        """Obtener predicción formateada."""
        probability = float(self.predict_proba(features.reshape(1, -1))[0][1])
        confidence = self._calculate_confidence(features)
        
        return MLPrediction(
            model_name=self.name,
            symbol=symbol,
            timeframe=timeframe,
            pattern_name=pattern_name,
            probability=probability,
            confidence=confidence,
            features_used=self._feature_names,
        )
```

---

## Feature Engineering

### Features para Patrones

```python
def extract_pattern_features(
    pattern: PatternResult,
    indicators: dict[str, float],
    candles: list[Candle],
) -> np.ndarray:
    """Extraer features para el modelo ML."""
    
    features = [
        # Features del patrón
        pattern.confidence,
        pattern.health,
        pattern.score,
        
        # Indicadores técnicos
        indicators.get("rsi", 50),
        indicators.get("macd", 0),
        indicators.get("macd_signal", 0),
        indicators.get("atr", 0),
        
        # EMAs
        indicators.get("ema_21", 0),
        indicators.get("ema_50", 0),
        indicators.get("ema_200", 0),
        
        # Volumen
        indicators.get("volume", 0),
        
        # Características del patrón
        _calculate_pattern_height(pattern),
        _calculate_pattern_duration(pattern),
        _calculate_volume_trend(candles),
    ]
    
    return np.array(features)

def _calculate_pattern_height(pattern: PatternResult) -> float:
    """Calcular altura del patrón como porcentaje."""
    if not pattern.key_levels:
        return 0.0
    
    values = list(pattern.key_levels.values())
    if len(values) < 2:
        return 0.0
    
    return (max(values) - min(values)) / min(values) * 100

def _calculate_pattern_duration(pattern: PatternResult) -> int:
    """Calcular duración del patrón en velas."""
    return pattern.current_candle_count

def _calculate_volume_trend(candles: list[Candle]) -> float:
    """Calcular tendencia del volumen."""
    if len(candles) < 10:
        return 0.0
    
    recent_vol = sum(c.data.volume for c in candles[-5:]) / 5
    avg_vol = sum(c.data.volume for c in candles[-20:]) / 20
    
    if avg_vol == 0:
        return 0.0
    
    return (recent_vol - avg_vol) / avg_vol
```

---

## Entrenamiento

### Flujo de Entrenamiento

```python
import asyncio
import numpy as np
from app.ml.models.random_forest import RandomForestModel
from app.database.base import get_async_session
from app.database.models import Pattern, Trade

async def train_model():
    # 1. Recopilar datos históricos
    async with get_async_session() as session:
        # Obtener patrones con resultados
        query = """
            SELECT 
                p.confidence,
                p.health,
                p.score,
                t.pnl,
                t.status
            FROM patterns p
            JOIN trades t ON p.pattern_uuid = t.pattern_name
            WHERE t.status = 'CLOSED'
        """
        # Ejecutar query y procesar datos
    
    # 2. Preparar features
    X = extract_features(patterns)
    y = np.array([1 if t.pnl > 0 else 0 for t in trades])
    
    # 3. Dividir datos
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 4. Entrenar modelo
    model = RandomForestModel(n_estimators=100, max_depth=10)
    metrics = model.train(X_train, y_train, feature_names=feature_names)
    
    print(f"Training accuracy: {metrics['train_accuracy']:.4f}")
    
    # 5. Evaluar
    eval_metrics = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"Precision: {eval_metrics['precision']:.4f}")
    print(f"Recall: {eval_metrics['recall']:.4f}")
    print(f"F1: {eval_metrics['f1']:.4f}")
    
    # 6. Guardar modelo
    model.save("models/rf_v1.pkl")
    
    return model

asyncio.run(train_model())
```

---

## Predicción en Tiempo Real

```python
from app.ml.base import MLPrediction
from app.patterns.base_pattern import PatternResult

async def predict_pattern_success(
    pattern: PatternResult,
    indicators: dict[str, float],
    candles: list[Candle],
) -> MLPrediction:
    """Predecir si un patrón tendrá éxito."""
    
    # Cargar modelo
    from app.ml.factory import MLModelFactory
    model = MLModelFactory.create("random_forest")
    model.load("models/rf_v1.pkl")
    
    # Extraer features
    features = extract_pattern_features(pattern, indicators, candles)
    
    # Obtener predicción
    prediction = model.get_prediction(
        features=features,
        symbol=pattern.symbol,
        timeframe=pattern.timeframe,
        pattern_name=pattern.pattern_name,
    )
    
    print(f"Probabilidad de éxito: {prediction.probability:.2%}")
    print(f"Confianza: {prediction.confidence:.2%}")
    print(f"Es accionable: {prediction.is_actionable}")
    
    return prediction
```

---

## Aprendizaje Continuo

### Offline Learning

Entrenamiento periódico con datos acumulados:

```python
from app.scheduler.main import Scheduler

async def offline_training_job():
    """Job de entrenamiento periódico."""
    print("Iniciando entrenamiento offline...")
    
    # Recopilar nuevos datos
    new_data = await collect_training_data()
    
    # Reentrenar modelo
    model = RandomForestModel()
    model.train(new_data.X, new_data.y)
    
    # Guardar versión
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f"models/rf_v{version}.pkl")
    
    print(f"Modelo guardado: rf_v{version}.pkl")

# Programar entrenamiento diario
scheduler = Scheduler()
await scheduler.add_interval(
    name="offline_training",
    func=offline_training_job,
    interval_seconds=86400,  # 24 horas
)
```

### Online Learning

Actualización con cada operación cerrada:

```python
async def on_trade_closed(trade: Trade, pattern: PatternResult):
    """Callback cuando se cierra una operación."""
    
    # Registrar resultado
    await store_training_sample(
        pattern=pattern,
        outcome=1 if trade.pnl > 0 else 0,
        pnl=trade.pnl,
    )
    
    # Actualizar estadísticas del modelo
    await update_model_metrics(pattern.pattern_name, trade.pnl > 0)
    
    # Si hay suficientes nuevos datos, reentrenar
    sample_count = await get_new_sample_count()
    if sample_count >= 100:
        await trigger_retraining()
```

---

## Métricas de Evaluación

### Métricas Disponibles

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

def evaluate_model(model, X_test, y_test):
    """Evaluar modelo completamente."""
    
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    
    return metrics
```

### Interpretación

| Métrica | Bueno | Aceptable | Malo |
|---------|-------|-----------|------|
| Accuracy | > 0.75 | 0.60-0.75 | < 0.60 |
| Precision | > 0.80 | 0.65-0.80 | < 0.65 |
| Recall | > 0.70 | 0.50-0.70 | < 0.50 |
| F1 | > 0.75 | 0.60-0.75 | < 0.60 |
| ROC AUC | > 0.80 | 0.65-0.80 | < 0.65 |

---

## Feature Importance

```python
# Obtener importancia de features
importance = model.get_feature_importance()

# Ordenar por importancia
sorted_importance = sorted(
    importance.items(),
    key=lambda x: x[1],
    reverse=True
)

print("Importancia de features:")
for feature, score in sorted_importance:
    print(f"  {feature}: {score:.4f}")
```

**Salida esperada**:

```
Importancia de features:
  confidence: 0.2341
  rsi: 0.1523
  macd: 0.1245
  volume_ratio: 0.0987
  atr: 0.0876
  ema_alignment: 0.0765
  pattern_height: 0.0654
  pattern_duration: 0.0543
```

---

## Guardado y Carga de Modelos

### Estructura de Archivos

```
models/
├── random_forest_v1.pkl
├── random_forest_v2.pkl
├── xgboost_v1.pkl
├── lstm_v1.pt
└── metadata/
    ├── rf_v1_metrics.json
    └── rf_v2_metrics.json
```

### Metadatos del Modelo

```json
{
  "model_name": "random_forest",
  "version": "v1",
  "trained_at": "2024-01-15T10:30:00Z",
  "training_samples": 10000,
  "features": ["confidence", "rsi", "macd", "atr", "volume"],
  "metrics": {
    "accuracy": 0.78,
    "precision": 0.82,
    "recall": 0.71,
    "f1": 0.76
  },
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 10
  }
}
```

---

## Ejemplo Completo

```python
import asyncio
import numpy as np
from datetime import datetime, timezone
from app.ml.models.random_forest import RandomForestModel
from app.patterns.base_pattern import PatternResult, PatternType
from app.market.candles.models import Candle, CandleData

async def ml_pipeline():
    """Pipeline completo de ML."""
    
    # 1. Crear modelo
    model = RandomForestModel(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    # 2. Generar datos de entrenamiento (ejemplo)
    np.random.seed(42)
    n_samples = 1000
    
    X_train = np.random.randn(n_samples, 10)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    
    feature_names = [
        "confidence", "health", "rsi", "macd", "macd_signal",
        "atr", "ema_21", "ema_50", "volume_ratio", "pattern_height"
    ]
    
    # 3. Entrenar
    print("Entrenando modelo...")
    train_metrics = model.train(X_train, y_train, feature_names=feature_names)
    print(f"Accuracy entrenamiento: {train_metrics['train_accuracy']:.4f}")
    
    # 4. Evaluar
    X_test = np.random.randn(200, 10)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)
    
    eval_metrics = model.evaluate(X_test, y_test)
    print(f"Accuracy test: {eval_metrics['accuracy']:.4f}")
    print(f"Precision: {eval_metrics['precision']:.4f}")
    print(f"Recall: {eval_metrics['recall']:.4f}")
    print(f"F1: {eval_metrics['f1']:.4f}")
    
    # 5. Guardar
    model.save("models/rf_demo.pkl")
    print("Modelo guardado")
    
    # 6. Cargar y predecir
    new_model = RandomForestModel()
    new_model.load("models/rf_demo.pkl")
    
    # Crear patrón de ejemplo
    pattern = PatternResult(
        pattern_name="double_top",
        pattern_type=PatternType.REVERSAL,
        symbol="BTCUSDT",
        timeframe="1h",
        confidence=0.85,
        health=90.0,
    )
    
    # Predecir
    features = np.array([0.85, 90, 72, 150, 120, 250, 51000, 50000, 1.3, 2.5])
    prediction = model.get_prediction(
        features=features,
        symbol="BTCUSDT",
        timeframe="1h",
        pattern_name="double_top",
    )
    
    print(f"\nPredicción:")
    print(f"  Probabilidad: {prediction.probability:.2%}")
    print(f"  Confianza: {prediction.confidence:.2%}")
    print(f"  Es accionable: {prediction.is_actionable}")

asyncio.run(ml_pipeline())
```

---

## Mejores Prácticas

1. **Datos suficientes**: Mínimo 1000 muestras para Random Forest
2. **Balance de clases**: Asegurar proporción similar de éxitos/fallos
3. **Validación cruzada**: Usar k-fold cross validation
4. **Evitar overfitting**: Limitar max_depth, usar early stopping
5. **Regularización**: Usar parámetros de regularización
6. **Monitoreo**: Rastrear degradación del modelo con el tiempo
7. **Versionado**: Guardar cada versión del modelo con métricas
8. **Reentrenamiento**: Actualizar periódicamente con nuevos datos
