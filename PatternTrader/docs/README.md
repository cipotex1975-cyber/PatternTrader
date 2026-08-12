# PatternTrader

**Plataforma profesional de detección de patrones chartistas con Machine Learning**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ¿Qué es PatternTrader?

PatternTrader es una plataforma de inteligencia para mercados financieros que detecta, evalúa y gestiona patrones chartistas automáticamente. Combina análisis técnico clásico con Machine Learning para generar señales de alta probabilidad.

### Características Principales

- **Detección de 20+ patrones chartistas** (Double Top, Head & Shoulders, Flags, Triangles, etc.)
- **Motor de estados** que administra el ciclo de vida de cada patrón
- **Sistema de scoring** que evalúa la calidad de cada detección
- **Machine Learning** para predicción de probabilidad de éxito
- **Aprendizaje continuo**: base de conocimiento alimentada por cada operación (offline y online)
- **Persistencia real**: PostgreSQL con migraciones Alembic, repositorios write-through (patrones, lifecycle, señales, trades, backtests, predicciones, modelos) y rehidratación del estado al arrancar
- **Backtesting avanzado** con motor independiente: simple, múltiple, walk-forward, Monte Carlo, out-of-sample, rolling window, validación cruzada, grid/random/búsqueda bayesiana
- **Métricas profesionales**: Win Rate, Profit Factor, Sharpe, Sortino, Calmar, Ulcer Index, Drawdown, Expectancy, Precision, Recall, F1 y Matriz de Confusión
- **Gestión de riesgo** automática
- **Notificaciones por Telegram** en tiempo real
- **API REST** completa con FastAPI
- **Dashboard** para visualización en tiempo real

---

## Arquitectura

El proyecto sigue los principios **Clean Architecture**, **SOLID** y **Domain Driven Design**:

```
PatternTrader/
├── app/
│   ├── core/           # Configuración, logging, eventos, excepciones
│   ├── data/           # Proveedores de datos, caché, WebSocket
│   ├── market/         # Motor de mercado (candles, indicadores)
│   ├── patterns/       # Detectores de patrones
│   ├── lifecycle/      # Motor de estados
│   ├── scoring/        # Sistema de puntuación
│   ├── confirmation/   # Motor de confirmación
│   ├── ml/             # Modelos de Machine Learning
│   ├── backtesting/    # Motor de backtesting
│   ├── risk/           # Gestión de riesgo
│   ├── signals/        # Generación de señales
│   ├── telegram/       # Notificaciones
│   ├── database/       # Modelos de base de datos
│   ├── datos_test/     # Archivos para trainin y test del modelo
│   ├── api/            # Endpoints REST
│   ├── strategy/       # Estrategias de trading
│   ├── optimizer/      # Optimización de parámetros
│   ├── visualization/  # Gráficos
│   ├── scheduler/      # Tareas programadas
│   └── monitor/        # Monitoreo del sistema
├── config/             # Archivos de configuración YAML
├── tests/              # Pruebas unitarias, integración, e2e
└── docs/               # Documentación
```

---

## Inicio Rápido

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/pattern-trader.git
cd pattern-trader

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -e ".[dev]"
```

### Configuración

```bash
# Editar la configuración principal
nano config/settings.yaml

# Configurar variables de entorno
export DB_PASSWORD="tu_password"
export TELEGRAM_BOT_TOKEN="tu_token"
export TELEGRAM_CHAT_ID="tu_chat_id"
```

### Ejecutar

```bash
# Iniciar el servidor
python -m app.main

# La API estará disponible en http://localhost:8000
```

---

## Ejemplos de Uso

### 1. Ejecutar el Pipeline Completo

```python
import asyncio
from app.patterns.pipeline import PatternPipeline

async def run_pipeline():
    # Usa el provider configurado (binance por defecto)
    pipeline = PatternPipeline()

    # Procesa el flujo completo para un símbolo/timeframe:
    # detección → lifecycle → health → confirmación → scoring
    #   → hipótesis → estrategia → señal → telegram
    stats = await pipeline.process_symbol("BTCUSDT", "1h")
    print(stats)  # tracked / active / expired / confirmed / signals_sent

asyncio.run(run_pipeline())
```

El pipeline también acepta un `data_source` inyectable (útil en tests):

```python
pipeline = PatternPipeline(data_source=lambda symbol, timeframe: my_candles)
await pipeline.process_symbol("BTCUSDT", "1h")
```

### 2. Detectar Patrones en un Activo

```python
import asyncio
from app.data.providers import DataProviderFactory
from app.market.candles.models import Candle, CandleData
from app.patterns.reversal.double_top import DoubleTopPattern
from datetime import datetime, timezone

async def detect_patterns():
    # Crear proveedor de datos (binance, bybit, yahoo, polygon,
    # alphavantage, metatrader, interactive_brokers)
    provider = DataProviderFactory.create("binance")
    await provider.connect()
    
    # Obtener datos históricos
    candles_data = await provider.get_history(
        symbol="BTCUSDT",
        timeframe="1h",
        limit=100
    )
    
    # Convertir a modelos internos
    candles = [
        Candle(
            symbol="BTCUSDT",
            timeframe="1h",
            data=CandleData(
                timestamp=c.timestamp,
                open=c.open,
                high=c.high,
                low=c.low,
                close=c.close,
                volume=c.volume,
            )
        )
        for c in candles_data
    ]
    
    # Detectar patrones
    detector = DoubleTopPattern()
    result = detector.detect(candles, "BTCUSDT", "1h")
    
    if result:
        print(f"Patrón detectado: {result.pattern_name}")
        print(f"Confianza: {result.confidence:.2%}")
        print(f"Niveles clave: {result.key_levels}")
    
    await provider.disconnect()

asyncio.run(detect_patterns())
```

### 3. Calcular Score de un Patrón

```python
from app.scoring.engine import ScoringEngine
from app.patterns.base_pattern import PatternResult, PatternType

# Crear un patrón de ejemplo
pattern = PatternResult(
    pattern_name="double_top",
    pattern_type=PatternType.REVERSAL,
    symbol="BTCUSDT",
    timeframe="1h",
    confidence=0.85,
    key_levels={"neckline": 50000, "peak1": 52000}
)

# Indicadores del mercado
indicators = {
    "rsi": 72,
    "macd": 150,
    "macd_signal": 120,
    "ema_21": 51000,
    "ema_50": 50500,
    "atr": 250,
    "volume": 1500000,
}

# Calcular score
engine = ScoringEngine()
score = engine.calculate_score(pattern, indicators)

print(f"Score total: {score.total_score}")
print(f"Grado: {score.grade}")
print(f"Confianza: {score.confidence:.2%}")
print(f"组件es:")
for component in score.components:
    print(f"  - {component.name}: {component.score:.0f}/100 (peso: {component.weight})")
```

### 4. Ejecutar Backtesting

```python
import asyncio
from app.backtesting.engine import BacktestEngine
from app.backtesting.models import BacktestConfig
from app.market.candles.models import Candle, CandleData
from app.patterns.base_pattern import PatternResult, PatternType, PatternStatus
from datetime import datetime, timezone, timedelta
import random

async def run_backtest():
    # Generar datos de ejemplo
    candles = []
    base_price = 50000
    for i in range(200):
        change = random.uniform(-500, 500)
        open_price = base_price + change
        high = open_price + random.uniform(0, 300)
        low = open_price - random.uniform(0, 300)
        close = open_price + random.uniform(-200, 200)
        
        candles.append(Candle(
            symbol="BTCUSDT",
            timeframe="1h",
            data=CandleData(
                timestamp=datetime.now(timezone.utc) - timedelta(hours=200-i),
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=random.randint(1000, 10000),
            )
        ))
        base_price = close
    
    # Crear patrones de ejemplo
    patterns = [
        PatternResult(
            pattern_name="double_bottom",
            pattern_type=PatternType.REVERSAL,
            symbol="BTCUSDT",
            timeframe="1h",
            confidence=0.8,
            status=PatternStatus.CONFIRMED,
            entry_price=50500,
            stop_loss=49500,
            take_profit=53000,
        )
    ]
    
    # Configurar y ejecutar backtest
    config = BacktestConfig(
        initial_capital=100000,
        risk_per_trade=0.02,
    )
    
    engine = BacktestEngine(config)
    result = engine.run(candles, patterns)
    
    # Mostrar resultados
    print(f"Trades totales: {result.metrics.total_trades}")
    print(f"Win Rate: {result.metrics.win_rate:.2%}")
    print(f"Profit Factor: {result.metrics.profit_factor:.2f}")
    print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.metrics.max_drawdown_pct:.2f}%")
    print(f"PnL Total: ${result.metrics.total_pnl:,.2f}")
    print(f"Retorno Total: {result.total_return:.2%}")

asyncio.run(run_backtest())
```

### 5. Evaluar Riesgo

```python
from app.risk.engine import RiskEngine
from app.patterns.base_pattern import PatternResult, PatternType

# Crear patrón
pattern = PatternResult(
    pattern_name="bull_flag",
    pattern_type=PatternType.CONTINUATION,
    symbol="ETHUSDT",
    timeframe="4h",
    confidence=0.88,
)

# Evaluar riesgo
engine = RiskEngine(initial_capital=100000)
assessment = engine.assess(
    pattern=pattern,
    entry_price=3000,
    stop_loss=2900,
    take_profit=3300,
)

print(f"Aceptable: {assessment.is_acceptable}")
print(f"Risk Score: {assessment.risk_score:.0f}/100")
print(f"R/R Ratio: {assessment.risk_reward_ratio:.2f}")
print(f"Tamaño posición: {assessment.position_size.size:.4f}")
print(f"Riesgo: ${assessment.position_size.risk_amount:,.2f}")
print(f"Recompensa potencial: ${assessment.position_size.potential_reward:,.2f}")

if assessment.warnings:
    print("\nAdvertencias:")
    for warning in assessment.warnings:
        print(f"  ⚠️  {warning}")

if assessment.recommendations:
    print("\nRecomendaciones:")
    for rec in assessment.recommendations:
        print(f"  💡 {rec}")
```

### 6. Usar la API REST

```bash
# Verificar salud del sistema
curl http://localhost:8000/api/v1/health

# Listar patrones disponibles
curl http://localhost:8000/api/v1/patterns/

# Obtener estadísticas de un patrón
curl http://localhost:8000/api/v1/patterns/double_top

# Ver señales activas
curl http://localhost:8000/api/v1/signals/

# Ver dashboard
curl http://localhost:8000/api/v1/dashboard/overview
```

---

## Patrones Soportados

### Patrones de Reversión
| Patrón | Nombre | Tipo | Confirmación |
|--------|--------|------|--------------|
| Double Top | `double_top` | Reversión bajista | 20 velas |
| Double Bottom | `double_bottom` | Reversión alcista | 20 velas |
| Head & Shoulders | `head_and_shoulders` | Reversión bajista | 25 velas |
| Inverse H&S | `inverse_head_and_shoulders` | Reversión alcista | 25 velas |

### Patrones de Continuación
| Patrón | Nombre | Tipo | Confirmación |
|--------|--------|------|--------------|
| Bull Flag | `bull_flag` | Continuación alcista | 12 velas |
| Bear Flag | `bear_flag` | Continuación bajista | 12 velas |
| Bull Pennant | `bull_pennant` | Continuación alcista | 12 velas |
| Bear Pennant | `bear_pennant` | Continuación bajista | 12 velas |
| Cup & Handle | `cup_and_handle` | Continuación alcista | 20 velas |

### Patrones Adicionales (Fase 5) — carpeta `app/patterns/neutral/`
| Patrón | Nombre | Tipo | Confirmación |
|--------|--------|------|--------------|
| Ascending Triangle | `ascending_triangle` | Neutral | 15 velas |
| Descending Triangle | `descending_triangle` | Neutral | 15 velas |
| Symmetric Triangle | `symmetric_triangle` | Neutral | 15 velas |
| Rising Wedge | `rising_wedge` | Neutral | 15 velas |
| Falling Wedge | `falling_wedge` | Neutral | 15 velas |
| Rectangle | `rectangle` | Neutral | 15 velas |
| Channel | `channel` | Neutral | 15 velas |
| Diamond | `diamond` | Neutral | 15 velas |
| Broadening | `broadening` | Neutral | 15 velas |
| Rounded Bottom | `rounded_bottom` | Reversión alcista | 25 velas |
| Triple Top | `triple_top` | Reversión bajista | 25 velas |
| Triple Bottom | `triple_bottom` | Reversión alcista | 25 velas |

---

## Configuración

Toda la configuración se administra en `config/settings.yaml`:

```yaml
# Ejemplo de configuración
scoring:
  weights:
    pattern_structure: 0.35
    volume: 0.20
    momentum: 0.10
    atr: 0.10
    rsi: 0.10
    macd: 0.05
    ema: 0.05
    ml_history: 0.05

risk:
  max_risk_per_trade: 0.02
  max_daily_risk: 0.06
  max_exposure_per_asset: 0.10

patterns:
  scoring:
    min_score_to_observe: 60
    min_score_to_prepare: 75
    min_score_to_alert: 85
    min_score_to_send: 95
```

Ver [CONFIGURATION.md](docs/CONFIGURATION.md) para más detalles.

---

## Documentación

| Documento | Descripción |
|-----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Arquitectura y diseño del sistema |
| [INSTALLATION.md](docs/INSTALLATION.md) | Guía de instalación detallada |
| [API.md](docs/API.md) | Documentación completa de la API REST |
| [PATTERNS.md](docs/PATTERNS.md) | Guía de detección de patrones |
| [MACHINE_LEARNING.md](docs/MACHINE_LEARNING.md) | Guía de modelos ML |
| [BACKTESTING.md](docs/BACKTESTING.md) | Guía de backtesting |
| [LEARNING.md](docs/LEARNING.md) | Aprendizaje continuo (offline y online) |
| [CONFIGURATION.md](docs/CONFIGURATION.md) | Referencia de configuración |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Guía de despliegue |
| [GAP_ANALYSIS.md](docs/GAP_ANALYSIS.md) | Auditoría de estado, gaps y roadmap de continuación |
| [plan_fase_4.md](docs/plan_fase_4.md) | Plan y ejecución de la Fase 4 (persistencia y API real) |

---

## Desarrollo

### Ejecutar Pruebas

```bash
# Todas las pruebas
pytest

# Con cobertura
pytest --cov=app --cov-report=html

# Solo pruebas unitarias
pytest tests/unit/

# Pruebas de integración
pytest tests/integration/
```

### Code Quality

```bash
# Formatear código
black app/ tests/
isort app/ tests/

# Verificar formato (sin modificar archivos)
black --check app/ tests/
isort --check-only app/ tests/

# Verificar tipos
mypy app/

# Linting (usa .flake8: max-line-length=100)
flake8 app/ tests/
```

### Entrenamiento Avanzado Multi-Modelo

```bash
# Entrenar y comparar los 9 modelos de ML para un par específico
python train_and_compare.py app/datos_test/USDCAD_H1_201005311000_202606010000.txt \
  --model all \
  --metric roc_auc \
  --db
```

1. Crear archivo en `app/patterns/reversal/` o `app/patterns/continuation/`
2. Heredar de `BasePattern`
3. Implementar métodos abstractos
4. Decorar con `@register_pattern`

```python
from app.patterns.base_pattern import BasePattern, PatternResult, PatternType
from app.patterns.registry import register_pattern

@register_pattern
class MyNewPattern(BasePattern):
    @property
    def name(self) -> str:
        return "my_new_pattern"
    
    @property
    def pattern_type(self) -> PatternType:
        return PatternType.REVERSAL
    
    @property
    def max_confirmation_candles(self) -> int:
        return 20
    
    def detect(self, candles, symbol, timeframe):
        # Tu lógica de detección aquí
        pass
    
    def validate(self, pattern, candles):
        # Tu lógica de validación aquí
        pass
```

---

## Roadmap

- [x] Estructura base del proyecto
- [x] Core modules (Config, Logger, Events)
- [x] Data Providers (Binance, Bybit, Yahoo, Polygon, AlphaVantage, MetaTrader, IB)
- [x] Market Engine (Candles, Indicators, Pivots, ZigZag, Fractals, Trendlines)
- [x] Pattern Detection (21 patrones)
- [x] Lifecycle Engine
- [x] Scoring System
- [x] Confirmation Engine
- [x] ML Framework (9 modelos: Random Forest, XGBoost, LightGBM, CatBoost, LSTM, Transformer, CNN, Isolation Forest, AutoEncoder)
- [x] Backtesting Engine (incluye Walk Forward, Monte Carlo, out-of-sample, rolling, CV, grid/random/bayesiana)
- [x] Risk Management
- [x] Signal Generation
- [x] Strategy Layer (hipótesis → estrategias → señal)
- [x] Telegram Integration
- [x] Database Models + persistencia (Alembic, repos write-through, rehidratación al arrancar)
- [x] REST API
- [ ] Dashboard UI completo

---

## Licencia

MIT License - Ver [LICENSE](LICENSE) para más detalles.

---

## Soporte

- **Issues**: [GitHub Issues](https://github.com/tu-usuario/pattern-trader/issues)
- **Docs**: [Documentación](docs/)
