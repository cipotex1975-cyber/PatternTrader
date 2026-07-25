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
│  │ Backtesting │  │     ML      │  │     Optimizer       │    │
│  │   Engine    │  │   Engine    │  │       Engine        │    │
│  └─────────────┘  └─────────────┘  └─────────────────────┘    │
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

**Uso**: Intercambiar algoritmos de detección y evaluación

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

**Ventaja**: Cada patrón encapsula su propia lógica de detección.

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
│   Scoring    │ ──→ Evalúa calidad del patrón
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Confirmation │ ──→ Valida antes de generar señal
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Signals    │ ──→ Genera señal de trading
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Telegram   │ ──→ Notifica al usuario
└──────────────┘
```

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
    def create_signal(self, pattern, score):
        # Solo genera la señal
        pass
```

---

## Extensibilidad

### Agregar un Nuevo Proveedor de Datos

1. Crear implementación de `IDataProvider`
2. Registrar en `DataProviderFactory`

```python
from app.data.providers.base import IDataProvider
from app.data.providers.factory import DataProviderFactory

class BybitProvider(IDataProvider):
    @property
    def name(self) -> str:
        return "bybit"
    
    async def connect(self):
        pass
    
    async def get_history(self, symbol, timeframe, start, end, limit):
        pass
    
    # ... otros métodos

# Registrar
DataProviderFactory.register("bybit", BybitProvider)
```

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
