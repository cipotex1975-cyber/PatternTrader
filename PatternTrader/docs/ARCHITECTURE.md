# Arquitectura de PatternTrader

## Visión General

PatternTrader está diseñado siguiendo los principios de **Clean Architecture**, **Domain Driven Design (DDD)** y **SOLID**. Esta arquitectura garantiza que el sistema sea:

- **Modular**: Cada componente es independiente y reemplazable
- **Testable**: Fácil de probar con mocks y fixtures
- **Extensible**: Nuevo código se agrega sin modificar el existente
- **Mantenible**: Separación clara de responsabilidades

---

## Principios de Diseño

### 1. Clean Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    UI / API Layer                        │
├─────────────────────────────────────────────────────────┤
│                 Interface Adapters                       │
├─────────────────────────────────────────────────────────┤
│                  Use Cases / Application                 │
├─────────────────────────────────────────────────────────┤
│                   Enterprise Rules                       │
│                    (Domain Layer)                        │
└─────────────────────────────────────────────────────────┘
```

**Regla de Dependencia**: Las dependencias solo apuntan hacia adentro. Las capas externas dependen de las internas, nunca al revés.

### 2. SOLID

| Principio | Aplicación |
|-----------|------------|
| **S**ingle Responsibility | Cada clase tiene una única razón para cambiar |
| **O**pen/Closed | Abierto a extensión, cerrado a modificación |
| **L**iskov Substitution | Los subtipos son substituibles por sus tiplos padre |
| **I**nterface Segregation | Interfaces pequeñas y específicas |
| **D**ependency Inversion | Depender de abstracciones, no de implementaciones |

### 3. DDD (Domain Driven Design)

- **Bounded Contexts**: Cada módulo define su propio dominio
- **Aggregates**: Patrones y Lifecycle son agregados con reglas de negocio
- **Domain Events**: Comunicación asíncrona entre módulos
- **Value Objects**: Candle, PatternResult, ScoreResult son inmutables

---

## Capas del Sistema

### Capa 1: Core (Reglas de Negocio)

```
app/core/
├── config/         # Configuración centralizada
├── logger/         # Sistema de logging
├── events/         # EventBus asíncrono
├── exceptions/     # Excepciones por dominio
├── constants/      # Constantes del sistema
└── utils/          # Utilidades generales
```

**Responsabilidades**:
- Definir interfaces y contratos
- Proporcionar servicios fundamentales
- No contener lógica de negocio específica

### Capa 2: Domain (Dominio)

```
app/market/         # Estructura del mercado
app/patterns/       # Detección de patrones
app/lifecycle/      # Ciclo de vida
app/health/         # Health score dinámico
app/scoring/        # Puntuación
app/confirmation/   # Confirmación
app/risk/           # Gestión de riesgo
app/signals/        # Generación de señales
```

**Responsabilidades**:
- Contener la lógica de negocio central
- Definir modelos de dominio
- Implementar reglas de negocio

### Capa 3: Application (Aplicación)

```
app/backtesting/    # Motor de backtesting
app/ml/             # Modelos de ML
app/strategy/       # Estrategias
app/optimizer/      # Optimización
```

**Responsabilidades**:
- Orquestar el uso de múltiples domain services
- Implementar casos de uso específicos
- Coordinar flujos de trabajo

### Capa 4: Infrastructure (Infraestructura)

```
app/data/           # Proveedores de datos
app/database/       # Persistencia
app/telegram/       # Notificaciones externas
app/api/            # Endpoints REST
```

**Responsabilidades**:
- Implementar interfaces definidas en capas inferiores
- Comunicar con sistemas externos
- Manejar persistencia y transporte

---

## Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer                               │
│                    (FastAPI Endpoints)                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                      Application Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐    │
│  │ Backtesting │  │     ML      │  │     Strategy        │    │
│  │   Engine    │  │   Engine    │  │       Engine        │    │
│  └─────────────┘  └─────────────┘  └─────────────────────┘    │
│  ┌─────────────┐                                               │
│  │  Optimizer  │                                               │
│  │   Engine    │                                               │
│  └─────────────┘                                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                        Domain Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐    │
│  │   Pattern   │  │  Lifecycle  │  │      Scoring        │    │
│  │   Engine    │  │   Engine    │  │       Engine        │    │
│  └─────────────┘  └─────────────┘  └─────────────────────┘    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐    │
│  │ Confirmation│  │    Risk     │  │      Signal         │    │
│  │   Engine    │  │   Engine    │  │       Engine        │    │
│  └─────────────┘  └─────────────┘  └─────────────────────┘    │
│  ┌─────────────┐                                              │
│  │   Health    │                                              │
│  │   Engine    │                                              │
│  └─────────────┘                                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    Infrastructure Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐    │
│  │    Data     │  │  Database   │  │     Telegram        │    │
│  │  Providers  │  │  (SQLAlchemy)│  │    Notifier         │    │
│  └─────────────┘  └─────────────┘  └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Patrones de Diseño Utilizados

### 1. Factory Pattern

**Uso**: Crear instancias de Data Providers y ML Models

```python
from app.data.providers.factory import DataProviderFactory
from app.ml.factory import MLModelFactory

# Crear proveedor
provider = DataProviderFactory.create("binance")

# Listar proveedores registrados
providers = DataProviderFactory.get_all()
# -> binance, bybit, yahoo, polygon, alphavantage,
#    metatrader, interactive_brokers

# Crear modelo ML
model = MLModelFactory.create("random_forest")
```

**Ventaja**: Fácil agregar nuevas implementaciones sin modificar código existente.

### 2. Registry Pattern

**Uso**: Registrar y descubrir patrones automáticamente

```python
from app.patterns.registry import register_pattern, PatternRegistry

@register_pattern
class MyPattern(BasePattern):
    pass

# Listar todos los patrones registrados
patterns = PatternRegistry.get_all()
```

**Ventaja**: Los patrones se auto-registran al importar el módulo.

### 3. Observer Pattern (Event Bus)

**Uso**: Comunicación asíncrona entre módulos

```python
from app.core.events.bus import get_event_bus
from app.core.events.models import Event, EventType

# Suscribirse a eventos
event_bus = get_event_bus()
event_bus.subscribe(EventType.PATTERN_DETECTED, my_handler)

# Publicar evento
await event_bus.publish(Event(
    type=EventType.PATTERN_DETECTED,
    source="PatternDetector",
    data={"pattern": "double_top"}
))
```

**Ventaja**: Desacoplamiento total entre productores y consumidores de eventos.

### 4. Strategy Pattern

**Uso**: Intercambiar algoritmos de detección y de decisión de trading

La capa de detección usa el patrón para intercambiar detectores:

```python
from app.patterns.base_pattern import BasePattern

class DoubleTopPattern(BasePattern):
    def detect(self, candles, symbol, timeframe):
        # Algoritmo específico para Double Top
        pass

class HeadAndShouldersPattern(BasePattern):
    def detect(self, candles, symbol, timeframe):
        # Algoritmo específico para Head & Shoulders
        pass
```

Desde la **Fase 2**, las estrategias de trading son también clases
intercambiables que consumen **hipótesis** (`PatternHypothesis`) y deciden
entrar o no. Se auto-registran como los patrones:

```python
from app.strategy.base import BaseStrategy
from app.strategy.registry import register_strategy

@register_strategy
class TrendFollowStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "trend_follow"

    def evaluate(self, hypothesis) -> StrategyDecision:
        # Decide ENTER (con StrategySignal) o NO_TRADE
        ...
```

El `StrategyEngine` (`app/strategy/engine.py`) ejecuta todas las estrategias
habilitadas sobre una hipótesis y elige la de mayor confianza. Así el pipeline
queda: patrón → **hipótesis** → **estrategia** → señal.

**Ventaja**: Cada patrón y cada estrategia encapsula su propia lógica, y añadir
una nueva no modifica el pipeline.

### 5. State Pattern (Lifecycle Engine)

**Uso**: Adminstrar transiciones de estado de patrones

```python
from app.lifecycle.engine import LifecycleEngine
from app.lifecycle.models import LifecycleState

engine = LifecycleEngine()

# Registrar patrón
lifecycle = await engine.register(pattern)

# Transicionar estado
await engine.transition(
    lifecycle.id,
    LifecycleState.CONFIRMED,
    reason="Breakout confirmado"
)
```

**Ventaja**: Transiciones de estado auditables y reversibles.

---

## Flujo de Datos

### Flujo Principal: Detección de Patrones

```
┌──────────────┐
│ Data Provider │ ──→ Obtiene datos del mercado
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Candle Store │ ──→ Almacena y organiza candles
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Indicators  │ ──→ Calcula indicadores técnicos
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│  Market Engine   │ ──→ Construye la estructura del mercado
│ (pivots, zigzag, │     (pivots, fractales, ZigZag, trendlines,
│  fractals, trend)│      canales y tendencia) en un MarketStructure
└──────┬───────────┘
       │
       ▼
┌──────────────┐
│   Patterns   │ ──→ Detecta patrones chartistas
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Lifecycle   │ ──→ Registra y administra estados
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Health     │ ──→ Recalcula salud (0-100) cada vela
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Scoring    │ ──→ Evalúa calidad del patrón
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Confirmation │ ──→ Valida antes de generar hipótesis
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Hipótesis    │ ──→ El pipeline emite PatternHypothesis
│ (pattern +   │     (pattern + indicadores + score + health
│  indicators) │      + confirmación), no señales directas
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Strategy    │ ──→ Las estrategias deciden ENTER/NO_TRADE
│  Engine      │     (TrendFollow, Breakout, Contrarian)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Signals    │ ──→ Genera señal de trading
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Telegram   │ ──→ Notifica al usuario (score ≥ 95)
└──────────────┘
```

Todo este flujo lo orquesta el **`PatternPipeline`**
(`app/patterns/pipeline.py`), ejecutado periódicamente por `PatternService`
(`app/patterns/service.py`) vía `Scheduler` al iniciar la API. El pipeline
**ya no genera señales directamente desde el patrón**: emite una hipótesis y
solo se crea la señal si alguna estrategia decide entrar. Las decisiones de
cada estrategia quedan en `result.metadata["strategy_decisions"]`.

### Flujo de Eventos

```
User ──→ API ──→ Application Service ──→ Domain Service ──→ Event Bus ──→ Handlers
                                                              │
                                                              ├──→ Lifecycle Handler
                                                              ├──→ Signal Handler
                                                              ├──→ ML Handler
                                                              └──→ Notification Handler
```

---

## Separación de Responsabilidades

### Ejemplo: Motor de Patrones

```python
# ❌ MAL: Un solo módulo hace todo
class PatternManager:
    def detect_pattern(self):
        pass
    
    def calculate_score(self):
        pass
    
    def send_signal(self):
        pass
    
    def update_database(self):
        pass

# ✅ BIEN: Cada módulo tiene una responsabilidad
class DoubleTopPattern(BasePattern):
    def detect(self, candles, symbol, timeframe):
        # Solo detecta el patrón
        pass

class ScoringEngine:
    def calculate_score(self, pattern, indicators):
        # Solo calcula el score
        pass

class SignalEngine:
    def create_signal(self, pattern, score, strategy_signal=None):
        # Solo genera la señal (con la decisión de la estrategia)
        pass

class TrendFollowStrategy(BaseStrategy):
    def evaluate(self, hypothesis):
        # Solo decide si entra o no con la hipótesis
        pass
```

---

## Extensibilidad

### Proveedores de Datos Implementados

| Proveedor | Módulo | Estado |
|-----------|--------|--------|
| Binance | `app/data/providers/binance/` | ✅ |
| Bybit | `app/data/providers/bybit/` | ✅ |
| Yahoo Finance | `app/data/providers/yahoo/` | ✅ |
| Polygon.io | `app/data/providers/polygon/` | ✅ |
| AlphaVantage | `app/data/providers/alphavantage/` | ✅ |
| MetaTrader 5 | `app/data/providers/metatrader/` | ✅ (dependencia opcional) |
| Interactive Brokers | `app/data/providers/interactive_brokers/` | ✅ (dependencia opcional) |

Todos implementan la interfaz `IDataProvider` y se registran automáticamente en `DataProviderFactory` al importar `app.data.providers`.

### Agregar un Nuevo Proveedor de Datos

1. Crear implementación de `IDataProvider` en `app/data/providers/<nombre>/`
2. Registrar en `DataProviderFactory`
3. Agregar la configuración en `app/core/config/settings.py` y `config/settings.yaml`

```python
from app.data.providers.base import IDataProvider
from app.data.providers.factory import DataProviderFactory

class KrakenProvider(IDataProvider):
    @property
    def name(self) -> str:
        return "kraken"
    
    async def connect(self):
        pass
    
    async def get_history(self, symbol, timeframe, start, end, limit):
        pass
    
    # ... otros métodos

# Registrar
DataProviderFactory.register("kraken", KrakenProvider)
```

> **Importante**: Al importar `app.data.providers`, los módulos registran sus proveedores automáticamente. Agregar un proveedor nunca modifica el resto del proyecto.

### Agregar un Nuevo Modelo ML

1. Crear implementación de `BaseMLModel`
2. Registrar en `MLModelFactory`

```python
from app.ml.base import BaseMLModel
from app.ml.factory import MLModelFactory

class XGBoostModel(BaseMLModel):
    @property
    def name(self) -> str:
        return "xgboost"
    
    def train(self, X, y, feature_names=None):
        pass
    
    def predict(self, X):
        pass
    
    # ... otros métodos

# Registrar
MLModelFactory.register("xgboost", XGBoostModel)
```

### Agregar un Nuevo Indicador

```python
from app.market.indicators.calculator import IndicatorCalculator

class ExtendedIndicatorCalculator(IndicatorCalculator):
    def calculate_custom_indicator(self, data):
        # Nuevo indicador
        pass
```

### Agregar una Nueva Estrategia

1. Crear una subclase de `BaseStrategy` en `app/strategy/strategies/`
2. Decorarla con `@register_strategy`
3. Opcional: añadir sus parámetros por defecto y la sección `strategies.params` en `config/settings.yaml`

```python
from app.strategy.base import BaseStrategy, StrategyDecision
from app.strategy.registry import register_strategy

@register_strategy
class MeanReversionStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "mean_reversion"

    def evaluate(self, hypothesis) -> StrategyDecision:
        # Lógica de decisión sobre la hipótesis
        return self._no_trade("sin señal por ahora")

    def get_parameters(self) -> dict:
        return {}

    def set_parameters(self, parameters: dict) -> None:
        pass
```

> **Importante**: Al importar `app.strategy`, las estrategias se auto-registran.
> Para activarla en runtime, añadir su nombre a `strategies.enabled` en `config/settings.yaml`.
> Las estrategias se pueden comparar sobre las mismas detecciones con
> `compare_strategies`/`run_strategy_backtest` de `app/strategy/evaluator.py`.

### Motor de Mercado (Market Engine)

El `MarketEngine` orquesta todos los detectores de estructura y agrupa su
resultado en un modelo `MarketStructure`:

| Módulo | Detecta | Modelos |
|--------|---------|---------|
| `app/market/pivots/` | Swing highs/lows | `Pivot`, `PivotType` |
| `app/market/fractals/` | Fractales de Bill Williams | `Fractal`, `FractalType` |
| `app/market/zigzag/` | Pivotes ZigZag (umbral %/ATR) | `ZigZagPoint`, `ZigZagType` |
| `app/market/trendlines/` | Trendlines, tendencia y canales | `Trendline`, `Trend`, `Channel` |
| `app/market/engine.py` | Orquesta los detectores | `MarketEngine`, `MarketStructure` |

```python
from app.market import MarketEngine

engine = MarketEngine()
structure = engine.analyze(candles, symbol="BTCUSDT", timeframe="1h")

structure.trend.direction          # uptrend | downtrend | sideways
structure.latest_indicators["rsi"] # Indicadores técnicos
structure.pivots                   # List[Pivot]
structure.zigzag                   # List[ZigZagPoint]
structure.trendlines               # List[Trendline]
structure.channels                 # List[Channel]
```

Cada detector puede usarse de forma independiente y acepta parámetros propios
que sobreescriben la configuración de `market.structure`:

```python
from app.market.pivots import PivotDetector
from app.market.trendlines import TrendlineDetector

pivots = PivotDetector(lookback=2).find_pivots(candles)
trendlines = TrendlineDetector(min_pivots=3).detect_from_pivots(candles, pivots)
```

---

## Concurrency y Performance

### AsyncIO

Todos los módulos de E/O usan `async/await`:

```python
async def get_data(self):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

### Procesamiento Paralelo

Para cálculos intensivos:

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

executor = ProcessPoolExecutor(max_workers=4)

async def process_multiple_symbols(symbols):
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(executor, process_symbol, symbol)
        for symbol in symbols
    ]
    return await asyncio.gather(*tasks)
```

### Caché

```python
from app.data.cache.memory import MemoryCache

cache = MemoryCache(default_ttl=300)

# Usar caché
data = cache.get("BTCUSDT:1h")
if data is None:
    data = await fetch_data()
    cache.set("BTCUSDT:1h", data, ttl=60)
```

---

## testing

### Estrategia de Pruebas

```
tests/
├── unit/           # Pruebas aisladas (80%)
├── integration/    # Pruebas de integración (15%)
└── e2e/           # Pruebas extremo a extremo (5%)
```

### Ejemplo de Prueba Unitaria

```python
import pytest
from app.patterns.reversal.double_top import DoubleTopPattern

def test_double_top_detection():
    pattern = DoubleTopPattern()
    assert pattern.name == "double_top"
    assert pattern.pattern_type.value == "reversal"
```

### Ejemplo de Prueba de Integración

```python
import pytest
from app.lifecycle.engine import LifecycleEngine
from app.patterns.base_pattern import PatternResult, PatternType

@pytest.mark.asyncio
async def test_lifecycle_workflow():
    engine = LifecycleEngine()
    pattern = PatternResult(
        pattern_name="double_top",
        pattern_type=PatternType.REVERSAL,
        symbol="BTCUSDT",
        timeframe="1h",
        confidence=0.85,
    )
    
    # Registrar
    lifecycle = await engine.register(pattern)
    assert lifecycle.current_state == LifecycleState.DETECTED
    
    # Transicionar
    await engine.transition(lifecycle.id, LifecycleState.FORMING)
    assert lifecycle.current_state == LifecycleState.FORMING
```

---

## Resumen

La arquitectura de PatternTrader está diseñada para:

1. **Crecer** con el proyecto sin degradarse
2. **Facilitar** pruebas automatizadas
3. **Permitir** cambios en componentes sin afectar otros
4. **Mantener** el código limpio y organizado
5. **Escalar** horizontalmente según la demanda

Cada componente puede ser desarrollado, testeado y desplegado independientemente, lo que permite trabajo en equipo eficiente y despliegues continuos.
