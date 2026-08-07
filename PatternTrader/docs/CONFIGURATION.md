# Guía de Configuración

## Visión General

PatternTrader utiliza archivos YAML para la configuración. Toda la configuración se encuentra en `config/settings.yaml`.

---

## Estructura del Archivo

```yaml
application:      # Configuración de la aplicación
server:           # Servidor web
database:         # Base de datos
logging:          # Sistema de logs
telegram:         # Notificaciones Telegram
data_providers:   # Proveedores de datos
market:           # Configuración del mercado
patterns:         # Configuración de patrones
strategies:       # Estrategias de trading
scoring:          # Sistema de puntuación
risk:             # Gestión de riesgo
backtesting:      # Motor de backtesting
ml:               # Modelos de ML
```

---

## Referencia Completa

### Application

```yaml
application:
  name: "PatternTrader"           # Nombre de la aplicación
  version: "0.1.0"                # Versión actual
  debug: false                    # Modo debug (true/false)
  environment: "development"      # Entorno: development/staging/production
```

### Server

```yaml
server:
  host: "0.0.0.0"                 # Host del servidor
  port: 8000                      # Puerto
  workers: 4                      # Número de workers
  reload: true                    # Auto-recarga en desarrollo
```

### Database

```yaml
database:
  host: "localhost"               # Host de PostgreSQL
  port: 5432                      # Puerto
  name: "pattern_trader"          # Nombre de la base de datos
  user: "postgres"                # Usuario
  password: "${DB_PASSWORD}"      # Contraseña (variable de entorno)
  pool_size: 20                   # Tamaño del pool de conexiones
  max_overflow: 10                # Conexiones extra
  echo: false                     # Log de queries SQL
```

### Logging

```yaml
logging:
  level: "INFO"                   # Nivel: DEBUG/INFO/WARNING/ERROR
  format: "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
  rotation: "100 MB"              # Rotación de archivos
  retention: "30 days"            # Retención de logs
  compression: "gz"               # Compresión
```

### Telegram

```yaml
telegram:
  bot_token: "${TELEGRAM_BOT_TOKEN}"  # Token del bot
  chat_id: "${TELEGRAM_CHAT_ID}"      # ID del chat
  enabled: false                       # Habilitar/deshabilitar
```

### Data Providers

```yaml
data_providers:
  default: "binance"              # Proveedor por defecto
  
  binance:
    api_key: "${BINANCE_API_KEY}"
    api_secret: "${BINANCE_API_SECRET}"
    testnet: true                 # Usar testnet
  
  bybit:
    api_key: "${BYBIT_API_KEY}"
    api_secret: "${BYBIT_API_SECRET}"
    testnet: true
  
  yahoo:
    enabled: true                 # Yahoo Finance (sin API key requerida)
  
  polygon:
    api_key: "${POLYGON_API_KEY}" # API key de Polygon.io
    enabled: true
  
  alphavantage:
    api_key: "${ALPHAVANTAGE_API_KEY}"  # API key de AlphaVantage
    enabled: true
  
  metatrader:
    enabled: false                # Requiere terminal MetaTrader 5 local
    login: 0                      # Login de la cuenta
    password: "${METATRADER_PASSWORD}"
    server: ""                    # Servidor del broker
    path: ""                      # Ruta del terminal MT5
  
  interactive_brokers:
    enabled: false                # Requiere IB Gateway o TWS con API
    host: "127.0.0.1"
    port: 7497                    # 7497=paper, 7496=live
    client_id: 1
```

### Proveedores Disponibles

| Proveedor | Clave | Fuente de datos | Dependencia |
|-----------|-------|-----------------|-------------|
| Binance | `binance` | Exchange (CCXT) | `ccxt` |
| Bybit | `bybit` | Exchange (CCXT) | `ccxt` |
| Yahoo Finance | `yahoo` | RestAPI de Yahoo | `yfinance` |
| Polygon.io | `polygon` | REST API | `httpx` |
| AlphaVantage | `alphavantage` | REST API | `httpx` |
| MetaTrader 5 | `metatrader` | Terminal MT5 local | `MetaTrader5` (opcional) |
| Interactive Brokers | `interactive_brokers` | IB Gateway/TWS | `ib_async` (opcional) |

**Notas**:
- `MetaTrader5` e `ib_async` son dependencias opcionales: instálalas solo si vas a usar esos proveedores (`pip install MetaTrader5`, `pip install ib_async`).
- La normalización de símbolos depende del proveedor: `BTCUSDT` → `BTC/USDT` (Bybit), `BTC-USD` (Yahoo), `X:BTCUSDT` (Polygon), `BTC` + `USDT` (AlphaVantage crypto), `BTCUSDT` (MT5/IB).
- AlphaVantage free tier limita a 25 requests/día; Polygon free tier ~5 requests/min.

### Market

```yaml
market:
  # Timeframes para análisis
  default_timeframes:
    - "1m"
    - "5m"
    - "15m"
    - "1h"
    - "4h"
    - "1d"
  
  # Símbolos por defecto
  default_symbols:
    - "BTCUSDT"
    - "ETHUSDT"
    - "BNBUSDT"
  
  # Configuración de indicadores
  indicators:
    ema_periods: [9, 21, 50, 100, 200]
    sma_periods: [20, 50, 100, 200]
    rsi_period: 14
    macd_fast: 12
    macd_slow: 26
    macd_signal: 9
    atr_period: 14
    bb_period: 20
    bb_std: 2.0
    vwap_enabled: true
    momentum_period: 10          # Período del momentum (ROC, en %)

  # Configuración de estructura de mercado (MarketEngine)
  structure:
    pivot_lookback: 2            # Barras a cada lado de un pivot swing
    fractal_window: 2            # Ventana de Bill Williams para fractales
    zigzag_threshold: 0.03       # Umbral % mínimo de retrazo del ZigZag
    zigzag_atr_multiplier: 1.5   # Multiplicador ATR para el umbral del ZigZag
    trend_min_pivots: 2          # Pivots mínimos para construir una trendline
    trend_strength_lookback: 5   # Ventana para medir la fuerza de la tendencia
    channel_slope_tolerance: 0.15  # Tolerancia relativa de pendientes (canales)
```

### Patterns

```yaml
patterns:
  # Configuración de scoring
  scoring:
    min_score_to_observe: 60      # Score mínimo para observar
    min_score_to_prepare: 75      # Score mínimo para preparar
    min_score_to_alert: 85        # Score mínimo para alertar
    min_score_to_send: 95         # Score mínimo para enviar señal
  
  # Configuración de lifecycle / pipeline
  lifecycle:
    enabled: true                 # Ejecuta el pipeline al iniciar la API
    check_interval_seconds: 5     # Intervalo de verificación
    max_patterns_per_symbol: 50   # Máximo de patrones por símbolo
    timeframes: ["15m", "1h", "4h"]  # Timeframes del pipeline
    candle_limit: 500             # Velas por símbolo en cada ciclo
  
  # Configuración de health
  health:
    recalculate_interval_seconds: 10  # Intervalo de recálculo
```

### Strategies

Las estrategias consumen las **hipótesis** del pipeline (patrón + indicadores +
score + health + confirmación) y deciden si entrar. Solo se genera una señal si
alguna estrategia decide `ENTER`.

```yaml
strategies:
  enabled:                    # Estrategias activas (en orden de evaluación)
    - "trend_follow"
    - "breakout"
    - "contrarian"
  params:                     # Parámetros por estrategia (opcional)
    trend_follow:
      min_score: 70.0         # Score mínimo del patrón para entrar
      min_health: 55.0        # Health mínimo
      default_size: 1.0       # Tamaño de posición por defecto
      momentum_threshold: 0.0 # Momentum mínimo (LONG) / máximo (SHORT)
    breakout:
      min_score: 70.0
      min_health: 50.0
      default_size: 1.0
      min_momentum: 0.0       # Momentum mínimo direccional
      rsi_min: 40.0           # Rango RSI para ruptura (LONG)
      rsi_max: 70.0
    contrarian:
      min_score: 60.0
      min_health: 50.0
      default_size: 0.5       # Tamaño reducido (riesgo mayor)
      oversold_rsi: 30.0      # RSI para considerar sobreventa (LONG)
      overbought_rsi: 70.0    # RSI para considerar sobrecompra (SHORT)
      max_reversal_momentum: 2.0  # Momentum máximo que "frena"
```

Estrategias disponibles (se auto-registran al importar `app.strategy`):

| Nombre | Clase | Lógica |
|--------|-------|--------|
| `trend_follow` | `TrendFollowStrategy` | Entra a favor de la tendencia (EMA9 vs EMA21 + momentum) |
| `breakout` | `BreakoutStrategy` | Entra en rupturas (momentum direccional + RSI en rango medio) |
| `contrarian` | `ContrarianStrategy` | Entra en reversales (RSI extremo + momentum perdiendo fuerza), solo `REVERSAL` |

Para añadir una estrategia, crear una subclase de `BaseStrategy` decorada con
`@register_strategy` y añadir su nombre a `enabled`.

### Scoring

```yaml
scoring:
  weights:
    pattern_structure: 0.35       # Peso de estructura del patrón
    volume: 0.20                  # Peso del volumen
    momentum: 0.10                # Peso del momentum
    atr: 0.10                     # Peso del ATR
    rsi: 0.10                     # Peso del RSI
    macd: 0.05                    # Peso del MACD
    ema: 0.05                     # Peso de las EMAs
    ml_history: 0.05              # Peso del historial ML
```

### Risk

```yaml
risk:
  max_risk_per_trade: 0.02        # 2% máximo por trade
  max_daily_risk: 0.06            # 6% máximo diario
  max_exposure_per_asset: 0.10    # 10% máximo por activo
  max_correlated_exposure: 0.15   # 15% máximo correlacionado
  default_rr_ratio: 2.0           # R/R ratio por defecto
  trailing_stop_enabled: false    # Trailing stop habilitado
```

### Backtesting

```yaml
backtesting:
  default_initial_capital: 100000     # Capital inicial
  default_commission: 0.001           # Comisión (0.1%)
  default_slippage: 0.0005            # Slippage (0.05%)
  walk_forward_splits: 5              # Divisiones walk-forward
  monte_carlo_simulations: 1000       # Simulaciones Monte Carlo
```

### Machine Learning

```yaml
ml:
  model_path: "./models/"             # Ruta de modelos
  training_data_path: "./data/training/"  # Ruta de datos
  
  models:
    random_forest:
      n_estimators: 100
      max_depth: 10
    
    xgboost:
      n_estimators: 100
      max_depth: 6
      learning_rate: 0.1
    
    lightgbm:
      n_estimators: 100
      max_depth: 6
      learning_rate: 0.1
    
    lstm:
      sequence_length: 60
      hidden_size: 128
      num_layers: 2
      dropout: 0.2
```

---

## Variables de Entorno

Las variables de entorno se usan para valores sensibles:

```bash
# Base de datos
export DB_PASSWORD="tu_password"

# Telegram
export TELEGRAM_BOT_TOKEN="tu_token"
export TELEGRAM_CHAT_ID="tu_chat_id"

# Proveedores
export BINANCE_API_KEY="tu_api_key"
export BINANCE_API_SECRET="tu_api_secret"
export BYBIT_API_KEY="tu_api_key"
export BYBIT_API_SECRET="tu_api_secret"
export POLYGON_API_KEY="tu_api_key"
export ALPHAVANTAGE_API_KEY="tu_api_key"
export METATRADER_PASSWORD="tu_password"
```

### Archivo .env

Crear archivo `.env` en la raíz del proyecto:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=pattern_trader
DB_USER=postgres
DB_PASSWORD=mi_password_segura

TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
TELEGRAM_CHAT_ID=-1001234567890

BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here

POLYGON_API_KEY=your_polygon_api_key
ALPHAVANTAGE_API_KEY=your_alphavantage_api_key

METATRADER_PASSWORD=your_mt5_password
```

---

## Acceder a la Configuración

### En Código Python

```python
from app.core.config.settings import get_settings

# Obtener configuración
settings = get_settings()

# Acceder a valores
print(settings.application.name)           # "PatternTrader"
print(settings.database.host)              # "localhost"
print(settings.risk.max_risk_per_trade)    # 0.02
print(settings.scoring.weights.volume)     # 0.20
```

### Ejemplos de Uso

```python
# Verificar si Telegram está habilitado
settings = get_settings()
if settings.telegram.enabled:
    print("Telegram habilitado")
else:
    print("Telegram deshabilitado")

# Obtener timeframes
timeframes = settings.market.default_timeframes
print(f"Timeframes: {timeframes}")

# Configurar indicadores
ema_periods = settings.market.indicators.ema_periods
print(f"EMAs: {ema_periods}")

# Límites de riesgo
risk_per_trade = settings.risk.max_risk_per_trade
print(f"Riesgo máximo por trade: {risk_per_trade:.2%}")
```

---

## Modos de Entorno

### Development

```yaml
application:
  environment: "development"
  debug: true

server:
  reload: true
  workers: 1

logging:
  level: "DEBUG"

database:
  echo: true  # Log queries
```

### Production

```yaml
application:
  environment: "production"
  debug: false

server:
  reload: false
  workers: 4

logging:
  level: "WARNING"

database:
  echo: false
  pool_size: 30
```

---

## Validación de Configuración

```python
from app.core.config.settings import Settings
from pydantic import ValidationError

try:
    settings = Settings(**config_data)
    print("Configuración válida")
except ValidationError as e:
    print(f"Error de configuración: {e}")
```

---

## Configuración por Defecto

Si no se especifica un valor, se usa el default:

| Sección | Parámetro | Default |
|---------|-----------|---------|
| server | port | 8000 |
| server | workers | 4 |
| database | pool_size | 20 |
| risk | max_risk_per_trade | 0.02 |
| scoring | min_score_to_send | 95 |

---

## Ejemplo Completo

```yaml
# config/settings.yaml personalizado

application:
  name: "MiPatternTrader"
  version: "1.0.0"
  debug: false
  environment: "production"

server:
  host: "0.0.0.0"
  port: 8080
  workers: 8
  reload: false

database:
  host: "db.example.com"
  port: 5432
  name: "pattern_trader_prod"
  user: "admin"
  password: "${DB_PASSWORD}"
  pool_size: 50
  max_overflow: 20
  echo: false

logging:
  level: "INFO"
  rotation: "500 MB"
  retention: "90 days"

telegram:
  bot_token: "${TELEGRAM_BOT_TOKEN}"
  chat_id: "${TELEGRAM_CHAT_ID}"
  enabled: true

data_providers:
  default: "binance"
  binance:
    api_key: "${BINANCE_API_KEY}"
    api_secret: "${BINANCE_API_SECRET}"
    testnet: false  # Production
  bybit:
    api_key: "${BYBIT_API_KEY}"
    api_secret: "${BYBIT_API_SECRET}"
    testnet: false
  yahoo:
    enabled: true
  polygon:
    api_key: "${POLYGON_API_KEY}"
    enabled: true
  alphavantage:
    api_key: "${ALPHAVANTAGE_API_KEY}"
    enabled: false  # Rate limit bajo (25 req/día)
  metatrader:
    enabled: false
    login: 0
    password: "${METATRADER_PASSWORD}"
    server: ""
    path: ""
  interactive_brokers:
    enabled: false
    host: "127.0.0.1"
    port: 7497
    client_id: 1

market:
  default_timeframes:
    - "15m"
    - "1h"
    - "4h"
    - "1d"
  
  default_symbols:
    - "BTCUSDT"
    - "ETHUSDT"
    - "BNBUSDT"
    - "SOLUSDT"
    - "ADAUSDT"
  
  indicators:
    ema_periods: [21, 50, 100, 200]
    rsi_period: 14
    atr_period: 14

patterns:
  scoring:
    min_score_to_observe: 65
    min_score_to_prepare: 78
    min_score_to_alert: 88
    min_score_to_send: 95
  
  lifecycle:
    check_interval_seconds: 3
    max_patterns_per_symbol: 30

scoring:
  weights:
    pattern_structure: 0.40
    volume: 0.25
    momentum: 0.15
    atr: 0.08
    rsi: 0.07
    macd: 0.03
    ema: 0.02
    ml_history: 0.00

risk:
  max_risk_per_trade: 0.015      # 1.5% más conservador
  max_daily_risk: 0.045          # 4.5% diario
  max_exposure_per_asset: 0.08   # 8% por activo
  default_rr_ratio: 2.5          # Mejor R/R

backtesting:
  default_initial_capital: 50000
  default_commission: 0.001
  default_slippage: 0.0003

ml:
  model_path: "/opt/pattern_trader/models/"
  training_data_path: "/opt/pattern_trader/data/training/"
```

---

## Solución de Problemas

### Error: `ValidationError`

**Causa**: Faltan campos requeridos o valores inválidos.

**Solución**:
```python
from pydantic import ValidationError

try:
    settings = Settings(**config)
except ValidationError as e:
    for error in e.errors():
        print(f"Campo: {error['loc']}")
        print(f"Error: {error['msg']}")
```

### Error: Variable de entorno no encontrada

**Causa**: No se ha definido la variable de entorno.

**Solución**:
```bash
# Verificar variable
echo $DB_PASSWORD

# Definir variable
export DB_PASSWORD="mi_password"

# O usar archivo .env
echo "DB_PASSWORD=mi_password" >> .env
```

### Error: Conexión a base de datos

**Causa**: Configuración incorrecta de database.

**Solución**:
```bash
# Verificar que PostgreSQL está corriendo
sudo systemctl status postgresql

# Verificar configuración
psql -h localhost -U postgres -d pattern_trader
```
