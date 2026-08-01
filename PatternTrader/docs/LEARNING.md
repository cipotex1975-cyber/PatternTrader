# Aprendizaje Continuo

## Visión General

PatternTrader aprende de cada operación ejecutada. Cada operación cerrada alimenta una **base de conocimiento** persistente que se usa para entrenar y mejorar modelos de predicción de forma continua.

```
Operación cerrada
     │
     ▼
KnowledgeEntry (instrumento, timeframe, patrón, variables,
               indicadores, resultado, drawdown, TP, SL, RR,
               duración, imagen, features)
     │
     ▼
Base de conocimiento (SQLAlchemy / memoria)
     │
     ├── Modo OFFLINE  → reentrena modelo sobre toda la base (validación cruzada)
     └── Modo ONLINE   → actualización incremental por operación (SGD)
```

## Base de Conocimiento

Cada registro (`KnowledgeEntry`) guarda exactamente lo siguiente:

| Campo | Descripción |
|-------|-------------|
| `instrument` | Instrumento (ej: BTCUSDT) |
| `timeframe` | Timeframe (ej: 1h, 4h) |
| `pattern` | Patrón detectado (ej: double_top) |
| `direction` | Dirección (LONG/SHORT) |
| `variables` | Variables del patrón (niveles, confianza, salud…) |
| `indicators` | Snapshot de indicadores (RSI, ATR, MACD, momentum…) |
| `outcome` | Resultado: WIN / LOSS / BREAKEVEN |
| `drawdown` | Máximo adverse excursion de la operación |
| `take_profit` / `stop_loss` | Niveles de TP y SL |
| `risk_reward` | Ratio riesgo/recompensa |
| `duration_seconds` | Duración de la operación |
| `image_path` | Ruta a captura/imagen del gráfico |
| `pnl` / `pnl_pct` | Resultado económico |
| `ml_features` | Vector de features generado |

## Modos de Aprendizaje

### Offline Learning

Reentrena el modelo (Random Forest) sobre **toda** la base de conocimiento y lo valida con **validación cruzada estratificada** (Precision, Recall, F1, ROC-AUC y Matriz de Confusión).

```python
from app.learning import LearningService, MemoryKnowledgeRepository, LearningMode

svc = LearningService(
    repository=MemoryKnowledgeRepository(),
    mode=LearningMode.OFFLINE,
    min_samples=10,       # mínimo de operaciones para empezar a entrenar
    retrain_every=10,     # reentrenar cada N operaciones
)
report = await svc.train_offline(n_splits=5)
# report["cross_validation"] -> {"average_f1": ..., "confusion_matrix": [...], ...}
```

### Online Learning

Actualiza el modelo de forma **incremental** con cada operación cerrada usando un `SGDClassifier` (`partial_fit`). No necesita el histórico completo.

```python
svc = LearningService(repository=MemoryKnowledgeRepository(), mode=LearningMode.ONLINE)
svc.set_mode(LearningMode.ONLINE)

# Cada operación cerrada se registra y alimenta el modelo al instante
entry = await svc.record_trade(trade, indicators={...}, variables={...})
```

## Registro Automático

El `LearningService` se suscribe al evento `TRADE_CLOSED` del event bus, de modo que **cada operación alimenta la base de conocimiento automáticamente**:

```python
from app.core.events.bus import get_event_bus
from app.core.events.models import Event, EventType

await svc.start()   # subscribe TRADE_CLOSED y TRADE_OPENED
await svc.stop()    # unsubscribe
```

## Predicción

```python
prediction = svc.predict(
    indicators={"rsi": 45, "atr": 100, "momentum": 1.2, ...},
    variables={"confidence": 0.8},
    instrument="BTCUSDT",
    timeframe="1h",
    pattern="double_top",
)
prediction.probability    # probabilidad de WIN (0-1)
prediction.confidence     # confianza de la predicción
```

Usa el mejor modelo disponible: online (si está entrenado) o offline.

## Estadísticas de la Base de Conocimiento

```python
stats = await svc.stats()
# {
#   "total_entries": 45,
#   "wins": 28, "losses": 17, "win_rate": 0.62,
#   "mode": "online",
#   "offline_report": {...},
#   "by_pattern": {"double_top": {"count": 12, "wins": 8, "pnl": 850.0}},
#   "by_instrument": {...},
# }
```

## Repositorios

| Repositorio | Persistencia | Uso |
|-------------|--------------|-----|
| `KnowledgeRepository` | SQLAlchemy async (`knowledge_entries`) | Producción |
| `MemoryKnowledgeRepository` | En memoria | Tests / demo |

## API REST

| Endpoint | Descripción |
|----------|-------------|
| `GET /api/v1/learning/entries` | Listar operaciones (filtros por instrumento/timeframe/patrón/resultado) |
| `GET /api/v1/learning/stats` | Estadísticas de la base de conocimiento |
| `POST /api/v1/learning/record` | Registrar una operación manualmente |
| `POST /api/v1/learning/train` | Entrenar offline con validación cruzada |
| `POST /api/v1/learning/predict` | Predecir probabilidad de éxito |
| `GET/POST /api/v1/learning/mode` | Consultar/cambiar modo (offline/online) |

## Tabla de Base de Datos

`knowledge_entries` se crea como parte de los modelos SQLAlchemy:

- `instrument`, `timeframe`, `pattern`, `direction`
- `variables` (JSON), `indicators` (JSON), `ml_features` (JSON)
- `outcome`, `pnl`, `pnl_pct`, `drawdown`
- `take_profit`, `stop_loss`, `risk_reward`, `duration_seconds`, `score`
- `entry_time`, `exit_time`, `image_path`, `created_at`
