# Cómo Empezar con PatternTrader

Guía rápida para activar el programa y configurar **varios proveedores de datos
activos a la vez** (por ejemplo, Binance para cripto y Yahoo Finance para forex).

---

## 1. Requisitos

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| Python | 3.11+ | 3.12 |
| PostgreSQL | 14+ | 16 |
| Git | 2.30+ | última versión |

---

## 2. Instalación

```bash
cd PatternTrader

# Entorno virtual
python -m venv venv
source venv/bin/activate   # Linux/macOS (o venv\Scripts\activate en Windows)

# Dependencias (dev incluye pytest, mypy, flake8, black, isort)
pip install -e ".[dev]"
```

> Los proveedores usan librerías ya incluidas (`ccxt`, `yfinance`, `httpx`).
> Solo `MetaTrader5` e `ib_async` son opcionales (`pip install MetaTrader5` /
> `pip install ib_async`) para esos proveedores específicos.

---

## 3. Base de datos (PostgreSQL)

```bash
sudo -u postgres createdb pattern_trader
sudo -u postgres psql -c "CREATE USER pattern_user WITH PASSWORD 'tu_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE pattern_trader TO pattern_user;"

# Aplicar migraciones
alembic upgrade head
```

---

## 4. Configurar `.env`

Crea o edita el archivo `.env` en la raíz del proyecto (la app lo lee solo):

```env
# --- Base de datos ---
DB_HOST=localhost
DB_PORT=5432
DB_NAME=pattern_trader
DB_USER=pattern_user
DB_PASSWORD=tu_password

# Opcional: URL completa (tiene prioridad si está exportada como variable real)
# DATABASE_URL=postgresql+asyncpg://pattern_user:tu_password@localhost:5432/pattern_trader

# --- Telegram (opcional) ---
TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghIkl
TELEGRAM_CHAT_ID=-1001234567890
TELEGRAM_ENABLED=true            # false para desactivar notificaciones

# --- Proveedores de datos ---
# Proveedor por defecto para símbolos SIN ruta explícita (ver sección 7)
DATA_PROVIDERS_DEFAULT=binance

BINANCE_API_KEY=tu_api_key
BINANCE_API_SECRET=tu_api_secret
# BINANCE_TESTNET=true

BYBIT_API_KEY=tu_api_key
BYBIT_API_SECRET=tu_api_secret

YAHOO_ENABLED=true              # Yahoo Finance (sin API key)

POLYGON_API_KEY=tu_api_key
ALPHAVANTAGE_API_KEY=tu_api_key
```

---

## 5. Activar el programa

```bash
python -m app.main
```

La aplicación:
1. Ejecuta `init_db()` (crea el esquema si no existe).
2. Arranca el bus de eventos, el servicio de aprendizaje y el `PatternService`.
3. Conecta los proveedores de datos configurados.
4. Programa el pipeline de patrones para cada símbolo × timeframe.

La API queda disponible en **http://localhost:8000**:

```bash
# Verificar salud del sistema
curl http://localhost:8000/api/v1/health

# Dashboard / señales / patrones
curl http://localhost:8000/api/v1/dashboard/overview
curl http://localhost:8000/api/v1/signals/
curl http://localhost:8000/api/v1/patterns/
```

Los logs se escriben en `logs/` (app y errores) y en pantalla.

> **Cadencia de validación**: el scheduler crea una tarea por símbolo ×
> timeframe. Cada tarea valida `vela / polling_checks_per_candle` (`1h`→60s,
> `4h`→240s por defecto) y el pipeline **solo descarga datos cuando hay una
> vela nueva**, reutilizando la caché entre ciclos. Ajustable en
> `patterns.lifecycle` de `config/settings.yaml` (ver [CONFIGURATION.md](CONFIGURATION.md)).

---

## 6. Comandos auxiliares

| Comando | Para qué sirve |
|---------|----------------|
| `python simulate_pipeline.py` | Simular el pipeline completo (detección → señal) sin servidor |
| `python run_backtest.py` | Ejecutar backtest sobre pares de `config/pairs.yaml` |
| `python train_and_compare.py <archivo.txt> --model all --metric roc_auc --db --promote` | Entrenar y comparar los modelos ML y promover el ganador |
| `python download_data.py` | Descargar datos OHLCV de Yahoo Finance (formato del backtest) |
| `pytest` | Ejecutar toda la suite de pruebas |

---

## 7. Varios proveedores activos a la vez

El sistema permite que **cada símbolo use un proveedor distinto** a través del
mapeo `market.symbol_providers` de `config/settings.yaml`. Los símbolos sin ruta
explícita usan el proveedor `DATA_PROVIDERS_DEFAULT`.

### Ejemplo: cripto por Binance + forex por Yahoo Finance

```yaml
# config/settings.yaml
market:
  default_symbols:
    - "BTCUSDT"
    - "ETHUSDT"
    - "EURUSD"
    - "USDJPY"

  # Routing símbolo -> proveedor
  symbol_providers:
    BTCUSDT: binance
    ETHUSDT: binance
    EURUSD: yahoo
    USDJPY: yahoo
```

Funcionamiento:
- `PatternService` recolecta los proveedores distintos del mapeo (más el
  default), los conecta una sola vez y registra un **resolver** en el pipeline
  (`set_provider_resolver`).
- El pipeline elige el proveedor por símbolo en cada ciclo
  (`pipeline._fetch_candles` → `_resolve_provider`).
- En `stop()` todos los proveedores se desconectan.

### Normalización de símbolos por proveedor

| Símbolo interno | Binance (CCXT) | Yahoo Finance |
|-----------------|----------------|---------------|
| `BTCUSDT` | `BTCUSDT` | `BTC-USD` |
| `ETHUSDT` | `ETHUSDT` | `ETH-USD` |
| `EURUSD` | — | `EURUSD=X` |
| `USDJPY` | — | `USDJPY=X` |

---

## 8. Limitaciones a tener en cuenta

- **Yahoo Finance** no ofrece el timeframe `4h` nativamente; el provider lo
  resuelve **agregando velas de `1h`** (agrupación de 4), así que puedes rutear
  forex a Yahoo y usar `4h` sin errores. Timeframes nativos soportados:
  `1m`, `2m`, `5m`, `15m`, `30m`, `1h`, `1d`, `1w`, `1M`.
- **Yahoo Finance intraday** solo ofrece historial de los últimos ~60 días.
  Para datos largos usa `1d`.
- **Yahoo no tiene order book** ni listing de símbolos (falla
  `get_order_book` / `get_symbols`); no afecta al pipeline de patrones.
- **Binance testnet** se activa con `BINANCE_TESTNET=true` (el default en
  código es `true`); desactívala para datos reales.
- Si un proveedor no puede conectarse, `PatternService` lo salta y solo los
  símbolos que dependían de él quedan sin datos (se avisa por log).

---

## 9. Solución de problemas

**El servidor no arranca / error de base de datos**
- Verifica que PostgreSQL esté corriendo y que `.env` tenga las `DB_*` correctas.
- Revisa `alembic current` y aplica migraciones (`alembic upgrade head`).

**Los símbolos de forex no devuelven datos (Yahoo)**
- Confirma que el símbolo esté mapeado a `yahoo` en `symbol_providers` (el
  timeframe `4h` ya funciona: se agrega desde `1h`).
- Comprueba el log: `tail -f logs/errors_$(date +%F).log`.

**ModuleNotFoundError: No module named 'app'**
- Ejecuta desde la raíz del repo o reinstala: `pip install -e .`

---

Ver también: [CONFIGURATION.md](CONFIGURATION.md) (referencia completa de
configuración), [INSTALLATION.md](INSTALLATION.md), [API.md](API.md),
[DEPLOYMENT.md](DEPLOYMENT.md).