# API REST - Documentación

## Visión General

PatternTrader expone una API REST construida con **FastAPI**. La API proporciona acceso a todas las funcionalidades del sistema de forma programática.

**Base URL**: `http://localhost:8000/api/v1`

**Documentación Interactiva**: `http://localhost:8000/docs` (Swagger UI)

**OpenAPI Schema**: `http://localhost:8000/openapi.json`

---

## Autenticación

Actualmente la API no requiere autenticación. En producción, se recomienda implementar:

- API Keys
- OAuth2
- JWT Tokens

---

## Endpoints

| Grupo | Prefijo | Descripción |
|-------|---------|-------------|
| Health | `/api/v1/health`, `/api/v1/info` | Estado del sistema |
| Patterns | `/api/v1/patterns` | Patrones disponibles, estadísticas |
| Signals | `/api/v1/signals` | Señales persistidas en DB |
| Strategies | `/api/v1/strategies` | Gestión runtime de estrategias (estado y parámetros) |
| Trades | `/api/v1/trades` | Trades persistidos en DB |
| Backtests | `/api/v1/backtests` | Resultados de backtests persistidos (incluye `equity_curve` y `trades` reales) |
| Learning | `/api/v1/learning` | Aprendizaje continuo (entries, stats, train, predict, mode) |
| Dashboard | `/api/v1/dashboard` | Resumen en tiempo real (estado compartido del pipeline) |
| Lifecycle | `/api/v1/lifecycle` | Ciclo de vida de patrones (rehidratado desde DB al arrancar) |
| Models | `/api/v1/models` | Modelos ML registrados y predicciones persistidas |

### Health Check

#### `GET /api/v1/health`

Verifica el estado del sistema.

**Respuesta**:

```json
{
  "status": "healthy",
  "application": "PatternTrader",
  "version": "0.1.0",
  "environment": "development"
}
```

#### `GET /api/v1/info`

Información general del sistema.

**Respuesta**:

```json
{
  "name": "PatternTrader",
  "version": "0.1.0",
  "debug": false
}
```

---

### Patterns

#### `GET /api/v1/patterns/`

Lista todos los patrones disponibles.

**Respuesta**:

```json
{
  "patterns": [
    {
      "name": "double_top",
      "type": "reversal",
      "max_confirmation_candles": 20
    },
    {
      "name": "double_bottom",
      "type": "reversal",
      "max_confirmation_candles": 20
    },
    {
      "name": "bull_flag",
      "type": "continuation",
      "max_confirmation_candles": 12
    }
  ]
}
```

#### `GET /api/v1/patterns/{pattern_name}`

Obtiene detalles de un patrón específico.

**Parámetros**:

| Nombre | Tipo | Descripción |
|--------|------|-------------|
| pattern_name | string | Nombre del patrón |

**Ejemplo**:

```bash
curl http://localhost:8000/api/v1/patterns/double_top
```

**Respuesta**:

```json
{
  "name": "double_top",
  "type": "reversal",
  "max_confirmation_candles": 20,
  "statistics": {
    "name": "double_top",
    "type": "reversal",
    "max_confirmation_candles": 20
  }
}
```

#### `GET /api/v1/patterns/{pattern_name}/statistics`

Obtiene estadísticas de un patrón.

**Ejemplo**:

```bash
curl http://localhost:8000/api/v1/patterns/bull_flag/statistics
```

**Respuesta**:

```json
{
  "name": "bull_flag",
  "type": "continuation",
  "max_confirmation_candles": 12
}
```

---

### Signals

#### `GET /api/v1/signals/`

Lista señales con filtros opcionales.

**Query Parameters**:

| Nombre | Tipo | Descripción |
|--------|------|-------------|
| status | string | Filtrar por estado (PENDING, SENT, DELIVERED, FAILED) |
| priority | string | Filtrar por prioridad (LOW, MEDIUM, HIGH, CRITICAL) |
| symbol | string | Filtrar por símbolo |

**Ejemplo**:

```bash
# Todas las señales
curl http://localhost:8000/api/v1/signals/

# Solo señales CRITICAL
curl "http://localhost:8000/api/v1/signals/?priority=CRITICAL"

# Señales de BTCUSDT
curl "http://localhost:8000/api/v1/signals/?symbol=BTCUSDT"
```

**Respuesta**:

```json
{
  "signals": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "symbol": "BTCUSDT",
      "pattern": "double_top",
      "direction": "SHORT",
      "priority": "HIGH",
      "status": "SENT",
      "score": 87.5,
      "entry_price": 52000.0,
      "stop_loss": 53000.0,
      "take_profit": 49000.0,
      "created_at": "2024-01-15T10:30:00Z"
    }
  ]
}
```

#### `GET /api/v1/signals/{signal_id}`

Obtiene detalles de una señal específica.

**Ejemplo**:

```bash
curl http://localhost:8000/api/v1/signals/550e8400-e29b-41d4-a716-446655440000
```

**Respuesta**:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "pattern": "double_top",
  "direction": "SHORT",
  "priority": "HIGH",
  "status": "SENT",
  "entry_price": 52000.0,
  "stop_loss": 53000.0,
  "take_profit": 49000.0,
  "risk_reward_ratio": 3.0,
  "score": 87.5,
  "health": 92.0,
  "ml_probability": 0.84,
  "reasons": [
    "Pattern: double_top detected",
    "Score: 87.5/100 (A-)",
    "ML probability: 84%",
    "volume: 85/100",
    "rsi: 78/100"
  ],
  "created_at": "2024-01-15T10:30:00Z"
}
```

---

### Strategies

Gestión **en caliente** de las estrategias registradas (`StrategyManager`).
Permite consultar el estado (habilitada/deshabilitada, parámetros activos) y
modificarlo sin reiniciar la API. Las estrategias se auto-registran al importar
`app.strategy`.

#### `GET /api/v1/strategies/`

Lista las estrategias registradas con su estado actual.

```bash
curl http://localhost:8000/api/v1/strategies
```

**Respuesta**:

```json
{
  "strategies": [
    {
      "name": "trend_follow",
      "description": "Entra en la dirección de la tendencia dominante",
      "enabled": true,
      "params": {"min_score": 80.0, "max_risk": 0.02}
    }
  ]
}
```

#### `GET /api/v1/strategies/{name}`

Detalle de una estrategia concreta.

```bash
curl http://localhost:8000/api/v1/strategies/trend_follow
```

> Devuelve `404` si el nombre no está registrado.

#### `PATCH /api/v1/strategies/{name}`

Activa/desactiva la estrategia o actualiza sus parámetros en caliente.

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `enabled` | bool | `true` para habilitarla, `false` para deshabilitarla |
| `params` | dict | Parámetros a actualizar (merge con los actuales) |

```bash
curl -X PATCH http://localhost:8000/api/v1/strategies/trend_follow \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'
```

```bash
curl -X PATCH http://localhost:8000/api/v1/strategies/trend_follow \
  -H "Content-Type: application/json" \
  -d '{"params": {"min_score": 85.0}}'
```

**Respuesta**: la estrategia con el estado/parámetros actualizados.

---

### Backtests

#### `GET /api/v1/backtests/`

Lista todos los backtests ejecutados.

**Respuesta**:

```json
{
  "backtests": [
    {
      "id": 0,
      "start_date": "2024-01-01T00:00:00Z",
      "end_date": "2024-01-15T00:00:00Z",
      "total_trades": 45,
      "win_rate": 0.62,
      "profit_factor": 2.1,
      "sharpe_ratio": 1.8,
      "total_pnl": 12500.0,
      "total_return": 0.125
    }
  ]
}
```

#### `GET /api/v1/backtests/{backtest_id}`

Obtiene detalles de un backtest específico.

**Ejemplo**:

```bash
curl http://localhost:8000/api/v1/backtests/0
```

**Respuesta**:

```json
{
  "id": 0,
  "config": {
    "initial_capital": 100000,
    "commission": 0.001,
    "slippage": 0.0005,
    "max_positions": 10,
    "risk_per_trade": 0.02
  },
  "metrics": {
    "total_trades": 45,
    "winning_trades": 28,
    "losing_trades": 17,
    "win_rate": 0.622,
    "profit_factor": 2.1,
    "sharpe_ratio": 1.8,
    "sortino_ratio": 2.2,
    "max_drawdown": 5200.0,
    "max_drawdown_pct": 5.2,
    "average_win": 850.0,
    "average_loss": -420.0,
    "expectancy": 277.8,
    "total_pnl": 12500.0,
    "total_pnl_pct": 12.5
  },
  "trades_count": 45,
  "equity_curve": [...],
  "start_date": "2024-01-01T00:00:00Z",
  "end_date": "2024-01-15T00:00:00Z",
  "initial_capital": 100000,
  "final_capital": 112500,
  "total_return": 0.125
}
```

#### `GET /api/v1/backtests/{backtest_id}/trades`

Obtiene las trades de un backtest.

**Ejemplo**:

```bash
curl http://localhost:8000/api/v1/backtests/0/trades
```

**Respuesta**:

```json
{
  "trades": [
    {
      "id": "trade_001",
      "symbol": "BTCUSDT",
      "direction": "LONG",
      "entry_price": 50000.0,
      "exit_price": 52500.0,
      "pnl": 2500.0,
      "pnl_pct": 0.05,
      "status": "CLOSED"
    }
  ]
}
```

---

### Trades

Trades persistidas en la tabla `trades` (motor de ejecución). Requieren una
base de datos configurada.

#### `GET /api/v1/trades/`

Lista trades con filtros opcionales.

```bash
curl "http://localhost:8000/api/v1/trades/?status=OPEN&symbol=BTCUSDT"
```

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `status` | `OPEN`/`CLOSED`/`CANCELLED` | Filtrar por estado |
| `symbol` | string | Filtrar por símbolo |

**Respuesta**:

```json
{
  "trades": [
    {
      "id": "a1b2c3d4-...",
      "symbol": "BTCUSDT",
      "timeframe": "1h",
      "direction": "LONG",
      "entry_price": 50000.0,
      "entry_time": "2026-08-07T10:00:00Z",
      "exit_price": null,
      "exit_time": null,
      "stop_loss": 49500.0,
      "take_profit": 51000.0,
      "size": 1.0,
      "pnl": 0.0,
      "pnl_pct": 0.0,
      "status": "OPEN",
      "pattern_name": "double_bottom"
    }
  ]
}
```

#### `GET /api/v1/trades/{trade_id}`

Detalle de un trade (UUID persistido).

```bash
curl http://localhost:8000/api/v1/trades/a1b2c3d4-...
```

**Respuesta**: igual que el listado pero con `score`, `duration_seconds` y
`metadata`.

---

### Lifecycle

Ciclo de vida de los patrones. Al reiniciar la API, el estado se rehidrata
desde la base de datos.

#### `GET /api/v1/lifecycle/statistics`

```bash
curl http://localhost:8000/api/v1/lifecycle/statistics
```

**Respuesta**:

```json
{
  "statistics": {
    "DETECTED": 5,
    "FORMING": 3,
    "WAITING_BREAKOUT": 2,
    "CONFIRMED": 1,
    "SIGNAL_SENT": 0,
    "OPEN": 0,
    "TP_HIT": 12,
    "SL_HIT": 8,
    "CLOSED": 45,
    "INVALIDATED": 15,
    "EXPIRED": 10,
    "CANCELLED": 2,
    "REJECTED": 3
  },
  "active": 11,
  "total": 106
}
```

#### `GET /api/v1/lifecycle/`

Lista de lifecycles con filtros opcionales.

```bash
curl "http://localhost:8000/api/v1/lifecycle/?state=OPEN&symbol=BTCUSDT&active=true"
```

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `state` | string | Filtrar por estado (DETECTED, FORMING, …) |
| `symbol` | string | Filtrar por símbolo |
| `active` | bool | Solo lifecycles activos |

#### `GET /api/v1/lifecycle/pattern/{pattern_id}`

Lifecycle de un patrón concreto (incluye transiciones).

#### `GET /api/v1/lifecycle/{lifecycle_id}`

Detalle de un lifecycle (incluye historial de transiciones).

---

### Learning

Aprendizaje continuo (offline/online).

#### `GET /api/v1/learning/entries`

Lista de entradas del conocimiento con filtros.

```bash
curl "http://localhost:8000/api/v1/learning/entries?pattern=double_top&limit=50"
```

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `instrument` | string | Filtrar por símbolo |
| `timeframe` | string | Filtrar por timeframe |
| `pattern` | string | Filtrar por patrón |
| `outcome` | string | Filtrar por resultado (WON/LOST/… ) |
| `limit` | int | Límite (default 100) |

#### `GET /api/v1/learning/stats`

Estadísticas del conocimiento (win rate por patrón/símbolo/timeframe).

#### `POST /api/v1/learning/record`

Registra una operación manual en la base de conocimiento.

```bash
curl -X POST http://localhost:8000/api/v1/learning/record \
  -H "Content-Type: application/json" \
  -d '{"trade": {"pattern": "double_top", "instrument": "BTCUSDT", "timeframe": "1h", "outcome": "WON"}, "indicators": {"rsi": 72, "atr": 250}}'
```

#### `POST /api/v1/learning/train`

Entrena el modelo offline sobre las entradas de conocimiento.

```bash
curl -X POST "http://localhost:8000/api/v1/learning/train?n_splits=5"
```

#### `POST /api/v1/learning/predict`

Predice la probabilidad de éxito con los indicadores dados (persiste la
predicción en `predictions`).

#### `GET/POST /api/v1/learning/mode`

Consulta o cambia el modo de aprendizaje (`OFFLINE`/`ONLINE`).

---

### AI Models

Modelos de Machine Learning registrados en `MLModelFactory` y persistidos en
la tabla `ml_models`.

#### `GET /api/v1/models/`

Lista modelos registrados y persistidos.

```bash
curl http://localhost:8000/api/v1/models
```

```json
{
  "registered": [
    {"name": "random_forest", "type": "classification", "loaded": true}
  ],
  "models": [
    {"name": "random_forest", "version": "1.0", "is_active": true, "metrics": {...}}
  ]
}
```

#### `GET /api/v1/models/{name}`

Detalle del modelo (trained, feature importance, registro en DB).

#### `POST /api/v1/models/{name}/predict`

Predice con el modelo entrenado y persiste el resultado en `predictions`.

```bash
curl -X POST http://localhost:8000/api/v1/models/random_forest/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [72, 250, 51000, 50500], "symbol": "BTCUSDT", "timeframe": "1h"}'
```

```json
{
  "model_name": "random_forest",
  "probability": 0.82,
  "confidence": 0.7,
  "features_used": ["rsi", "atr", "ema_21", "ema_50"]
}
```

> Devuelve `404` si el modelo no existe y `400` si no está entrenado.

---

### Dashboard

#### `GET /api/v1/dashboard/overview`

Resumen general del dashboard.

**Respuesta**:

```json
{
  "statistics": {
    "DETECTED": 5,
    "FORMING": 3,
    "WAITING_BREAKOUT": 2,
    "CONFIRMED": 1,
    "SIGNAL_SENT": 0,
    "OPEN": 0,
    "TP_HIT": 12,
    "SL_HIT": 8,
    "CLOSED": 45,
    "INVALIDATED": 15,
    "EXPIRED": 10,
    "CANCELLED": 2,
    "REJECTED": 3
  },
  "active_patterns": 11,
  "total_lifecycles": 106
}
```

#### `GET /api/v1/dashboard/active`

Lista patrones activos.

**Respuesta**:

```json
{
  "patterns": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "symbol": "BTCUSDT",
      "timeframe": "1h",
      "pattern": "double_top",
      "state": "FORMING",
      "transitions": 2,
      "created_at": "2024-01-15T10:30:00Z"
    }
  ]
}
```

#### `GET /api/v1/dashboard/by-state/{state}`

Filtra patrones por estado.

**Estados disponibles**:

- `DETECTED`
- `FORMING`
- `WAITING_BREAKOUT`
- `CONFIRMED`
- `SIGNAL_SENT`
- `OPEN`
- `TP_HIT`
- `SL_HIT`
- `CLOSED`
- `INVALIDATED`
- `EXPIRED`
- `CANCELLED`
- `REJECTED`

**Ejemplo**:

```bash
curl http://localhost:8000/api/v1/dashboard/by-state/CONFIRMED
```

**Respuesta**:

```json
{
  "state": "CONFIRMED",
  "count": 3,
  "patterns": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "symbol": "BTCUSDT",
      "timeframe": "1h",
      "pattern": "double_top",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ]
}
```

#### `GET /api/v1/dashboard/by-symbol/{symbol}`

Filtra patrones por símbolo.

**Ejemplo**:

```bash
curl http://localhost:8000/api/v1/dashboard/by-symbol/BTCUSDT
```

---

## Códigos de Respuesta

| Código | Descripción |
|--------|-------------|
| 200 | Éxito |
| 400 | Solicitud inválida |
| 404 | Recurso no encontrado |
| 422 | Error de validación |
| 500 | Error interno del servidor |

---

## Ejemplos con Python

### Usando `httpx`

```python
import httpx
import asyncio

async def get_patterns():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/api/v1/patterns/")
        return response.json()

async def get_signals(priority: str = None):
    async with httpx.AsyncClient() as client:
        params = {}
        if priority:
            params["priority"] = priority
        response = await client.get(
            "http://localhost:8000/api/v1/signals/",
            params=params
        )
        return response.json()

# Ejecutar
patterns = asyncio.run(get_patterns())
signals = asyncio.run(get_signals(priority="CRITICAL"))
```

### Usando `requests`

```python
import requests

def get_patterns():
    response = requests.get("http://localhost:8000/api/v1/patterns/")
    return response.json()

def get_backtest(backtest_id: int):
    response = requests.get(f"http://localhost:8000/api/v1/backtests/{backtest_id}")
    return response.json()

# Ejecutar
patterns = get_patterns()
backtest = get_backtest(0)
```

### Usando `aiohttp`

```python
import aiohttp
import asyncio

async def get_patterns():
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8000/api/v1/patterns/") as response:
            return await response.json()

async def main():
    patterns = await get_patterns()
    print(patterns)

asyncio.run(main())
```

---

## WebSocket (Próximamente)

La API soportará WebSocket para datos en tiempo real:

```python
import websockets
import json

async def connect_websocket():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        while True:
            message = await ws.recv()
            data = json.loads(message)
            print(f"Nuevo evento: {data['type']}")

asyncio.run(connect_websocket())
```

---

## Errores Comunes

### 422 Validation Error

```json
{
  "detail": [
    {
      "loc": ["body", "symbol"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Solución**: Verificar que todos los campos requeridos están presentes.

### 404 Not Found

```json
{
  "detail": "Pattern not found"
}
```

**Solución**: Verificar que el nombre del patrón es correcto.

---

## Rate Limiting

Actualmente no hay rate limiting implementado. En producción, se recomienda:

- 100 requests por minuto por IP
- 1000 requests por hora por API key

---

## Versión de la API

La API sigue versionado semántico:

- **v1**: Versión actual (experimental)
- **v2**: Próxima versión estable

Los endpoints pueden cambiar sin previo aviso en v1.
