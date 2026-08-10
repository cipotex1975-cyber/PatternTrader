# Guía de Patrones Chartistas

## Visión General

PatternTrader detecta y evalúa patrones chartistas clásicos utilizando análisis técnico cuantitativo. Cada patrón es una clase independiente que implementa una interfaz común.

---

## Patrones Disponibles

### Patrones de Reversión

#### 1. Double Top (`double_top`)

**Descripción**: Patrón bajista que indica el fin de una tendencia alcista. El precio alcanza un máximo dos veces con un retroceso intermedio.

```
    ┌───┐     ┌───┐
    │   │     │   │
    │   │     │   │
────┘   └─────┘   └──── Neckline
              ↓
         Target
```

**Características**:
- **Tipo**: Reversión bajista
- **Confirmación**: 20 velas
- **Señal**: Ruptura del neckline hacia abajo

**Ejemplo de uso**:

```python
from app.patterns.base_pattern import BasePattern, PatternResult, PatternType, TradeDirection
from app.market.candles.models import Candle, CandleData
from datetime import datetime, timezone

# Crear detector
detector = DoubleTopPattern()

# Datos de ejemplo (100 velas)
candles = [
    Candle(
        symbol="BTCUSDT",
        timeframe="1h",
        data=CandleData(
            timestamp=datetime.now(timezone.utc),
            open=50000 + i * 10,
            high=50100 + i * 10,
            low=49900 + i * 10,
            close=50050 + i * 10,
            volume=1000 + i * 100,
        )
    )
    for i in range(100)
]

# Detectar patrón
result = detector.detect(candles, "BTCUSDT", "1h")

if result:
    print(f"Patrón detectado: {result.pattern_name}")
    print(f"Dirección: {result.direction.value}")  # SHORT para double_top
    print(f"Confianza: {result.confidence:.2%}")
    print(f"Niveles clave:")
    for level, price in result.key_levels.items():
        print(f"  {level}: ${price:,.2f}")
```

**Niveles Clave**:
- `peak1`: Primer máximo
- `peak2`: Segundo máximo
- `neckline`: Línea de soporte (retroceso mínimo)
- `target`: Objetivo de precio

---

#### 2. Double Bottom (`double_bottom`)

**Descripción**: Patrón alcista que indica el fin de una tendencia bajista. El precio alcanza un mínimo dos veces con un rebote intermedio.

```
────┐   ┌─────┐   ┌──── Neckline
    │   │     │   │
    │   │     │   │
    └───┘     └───┘
              ↑
         Target
```

**Características**:
- **Tipo**: Reversión alcista
- **Confirmación**: 20 velas
- **Señal**: Ruptura del neckline hacia arriba

**Ejemplo**:

```python
from app.patterns.reversal.double_bottom import DoubleBottomPattern

detector = DoubleBottomPattern()
result = detector.detect(candles, "BTCUSDT", "1h")

if result:
    print(f"Target alcista: ${result.key_levels['target']:,.2f}")
    print(f"Stop Loss sugerido: ${result.key_levels['trough1'] - 100:,.2f}")
```

---

#### 3. Head and Shoulders (`head_and_shoulders`)

**Descripción**: Patrón bajista con tres picos: el central es el más alto (cabeza), y los laterales son más bajos (hombros).

```
        ┌───┐
   ┌───┐│   │┌───┐
   │   ││   ││   │
   │   │└───┘│   │
───┘   └─────┘   └─── Neckline
              ↓
         Target
```

**Características**:
- **Tipo**: Reversión bajista
- **Confirmación**: 25 velas
- **Señal**: Ruptura del neckline

**Ejemplo**:

```python
from app.patterns.reversal.head_and_shoulders import HeadAndShouldersPattern

detector = HeadAndShouldersPattern()
result = detector.detect(candles, "BTCUSDT", "1h")

if result:
    print(f"Cabeza: ${result.key_levels['head']:,.2f}")
    print(f"Hombro izquierdo: ${result.key_levels['left_shoulder']:,.2f}")
    print(f"Hombro derecho: ${result.key_levels['right_shoulder']:,.2f}")
    print(f"Neckline: ${result.key_levels['neckline']:,.2f}")
    print(f"Target: ${result.key_levels['target']:,.2f}")
```

---

#### 4. Inverse Head and Shoulders (`inverse_head_and_shoulders`)

**Descripción**: Patrón alcista inverso del Head and Shoulders. Tres mínimos, el central es el más bajo.

```
───┐   ┌─────┐   ┌─── Neckline
   │   │     │   │
   │   ┌───┐ │   │
   └───┘   └─┘   └───
              ↑
         Target
```

**Características**:
- **Tipo**: Reversión alcista
- **Confirmación**: 25 velas
- **Señal**: Ruptura del neckline hacia arriba

---

### Patrones de Continuación

#### 5. Bull Flag (`bull_flag`)

**Descripción**: Patrón alcista que forma una bandera después de un movimiento fuerte hacia arriba.

```
     │
     │ Pole
     │
     └────┐
          │ Flag
     ┌────┘
     │
     │ Continuación
     ↓
```

**Características**:
- **Tipo**: Continuación alcista
- **Confirmación**: 12 velas
- **Señal**: Ruptura de la bandera hacia arriba

**Ejemplo**:

```python
from app.patterns.continuation.bull_flag import BullFlagPattern

detector = BullFlagPattern()
result = detector.detect(candles, "BTCUSDT", "1h")

if result:
    print(f"Polo alto: ${result.key_levels['pole_high']:,.2f}")
    print(f"Flag bajo: ${result.key_levels['flag_low']:,.2f}")
    print(f"Target: ${result.key_levels['target']:,.2f}")
```

---

#### 6. Bear Flag (`bear_flag`)

**Descripción**: Patrón bajista que forma una bandera después de un movimiento fuerte hacia abajo.

```
     │
     │ Continuación
     │
     ┌────┐
          │ Flag
     └────┐
          │ Pole
          │
          ↓
```

**Características**:
- **Tipo**: Continuación bajista
- **Confirmación**: 12 velas
- **Señal**: Ruptura de la bandera hacia abajo

---

#### 7. Bull Pennant (`bull_pennant`)

**Descripción**: Similar a Bull Flag pero con líneas convergentes formando un triángulo.

```
     │
     │ Pole
     │
     └───┐
         ╲ Flag
          ╲
     ╱────┘
     │
     ↓
```

**Características**:
- **Tipo**: Continuación alcista
- **Confirmación**: 12 velas
- **Señal**: Ruptura del triángulo hacia arriba

---

#### 8. Bear Pennant (`bear_pennant`)

**Descripción**: Similar a Bear Flag pero con líneas convergentes.

```
     │
     ╱────┐
          ╱ Flag
         ╱
     └───┘
     │ Pole
     │
     ↓
```

**Características**:
- **Tipo**: Continuación bajista
- **Confirmación**: 12 velas
- **Señal**: Ruptura del triángulo hacia abajo

---

### Patrones adicionales (Fase 5) — carpeta `app/patterns/neutral/`

Los 13 patrones nuevos se implementan en `app/patterns/neutral/` con geometría compartida en `neutral/geometry.py` (`fit_line`, `line_at`, `find_peaks`, `find_troughs`). Usan regresión lineal sobre highs/lows para identificar estructura:

| Patrón | Tipo | Confirmación | Detección clave |
|--------|------|--------------|-----------------|
| `ascending_triangle` | Neutral | 15 velas | Resistencia plana + soporte ascendente convergente |
| `descending_triangle` | Neutral | 15 velas | Soporte plano + resistencia descendente convergente |
| `symmetric_triangle` | Neutral | 15 velas | Resistencia descendente + soporte ascendente |
| `rising_wedge` | Neutral | 15 velas | Ambas líneas suben, soporte más inclinado (converge) |
| `falling_wedge` | Neutral | 15 velas | Ambas líneas bajan, resistencia más inclinada (converge) |
| `rectangle` | Neutral | 15 velas | Resistencia y soporte horizontales y paralelos |
| `channel` | Neutral | 15 velas | Líneas paralelas con pendiente (no planas) |
| `cup_and_handle` | Continuación | 20 velas | Copa en U (bordes a la misma altura) + asa tras el borde |
| `rounded_bottom` | Reversión | 25 velas | Parábola convexa (coef. cuadrático > 0) en los lows |
| `diamond` | Neutral | 15 velas | Rango se contrae y luego se expande |
| `broadening` | Neutral | 15 velas | Resistencia sube y soporte baja (rango creciente) |
| `triple_top` | Reversión | 25 velas | 3 picos a nivel similar con neckline en los valles |
| `triple_bottom` | Reversión | 25 velas | 3 valles a nivel similar con neckline en los picos |

Los `key_levels` de los patrones nuevos son compatibles con `_prepare_price_levels` del pipeline (`neckline`, `support`, `valley`, `target`, `peak1..3`, `trough1..3`, etc.), por lo que generan entry/stop/take-profit automáticamente.

---

## Interfaz de Patrones

Todos los patrones heredan de `BasePattern`:

```python
from app.patterns.base_pattern import BasePattern, PatternResult, PatternType, TradeDirection

class BasePattern(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre del patrón."""
        ...
    
    @property
    @abstractmethod
    def pattern_type(self) -> PatternType:
        """Tipo: REVERSAL, CONTINUATION, o NEUTRAL."""
        ...
    
    @property
    @abstractmethod
    def max_confirmation_candles(self) -> int:
        """Máximo de velas para confirmar."""
        ...
    
    @abstractmethod
    def detect(self, candles, symbol, timeframe) -> Optional[PatternResult]:
        """Detectar el patrón en las velas."""
        ...
    
    @abstractmethod
    def validate(self, pattern, candles) -> bool:
        """Validar si el patrón sigue vigente."""
        ...
    
    def update(self, pattern, candles) -> PatternResult:
        """Actualizar estado del patrón con nuevas velas."""
        ...
    
    def invalidate(self, pattern, reason="") -> PatternResult:
        """Invalidar el patrón."""
        ...
    
    def statistics(self) -> dict:
        """Obtener estadísticas del patrón."""
        ...
    
    def plot(self, candles, pattern=None, title=None):
        """Generar figura plotly con el patrón sobre las velas."""
        ...
```

### Campo `direction` (TradeDirection)

Cada patrón tiene un campo `direction` que indica la dirección del trade:

```python
class TradeDirection(str, Enum):
    LONG = "LONG"    # Comprar (patrón alcista)
    SHORT = "SHORT"  # Vender (patrón bajista)
```

**Dirección por patrón:**

| Patrón | Dirección | Razón |
|--------|-----------|-------|
| `double_top` | SHORT | Reversión bajista |
| `double_bottom` | LONG | Reversión alcista |
| `head_and_shoulders` | SHORT | Reversión bajista |
| `inverse_head_and_shoulders` | LONG | Reversión alcista |
| `bull_flag` | LONG | Continuación alcista |
| `bear_flag` | SHORT | Continuación bajista |
| `bull_pennant` | LONG | Continuación alcista |
| `bear_pennant` | SHORT | Continuación bajista |
| `ascending_triangle` | LONG | Triángulo ascendente (ruptura alcista) |
| `descending_triangle` | SHORT | Triángulo descendente (ruptura bajista) |
| `symmetric_triangle` | LONG | Triángulo simétrico |
| `rising_wedge` | SHORT | Cuña ascendente (ruptura bajista) |
| `falling_wedge` | LONG | Cuña descendente (ruptura alcista) |
| `rectangle` | LONG | Rectángulo (continuación) |
| `channel` | LONG | Canal paralelo |
| `cup_and_handle` | LONG | Copa con asa (continuación alcista) |
| `rounded_bottom` | LONG | Suelo redondeado (reversión alcista) |
| `diamond` | LONG | Diamante |
| `broadening` | LONG | Megáfono / broadening |
| `triple_top` | SHORT | Triple techo (reversión bajista) |
| `triple_bottom` | LONG | Triple suelo (reversión alcista) |

**Ejemplo de uso:**

```python
from app.patterns.base_pattern import PatternResult, TradeDirection

# Crear patrón con dirección
pattern = PatternResult(
    pattern_name="double_top",
    pattern_type=PatternType.REVERSAL,
    symbol="BTCUSDT",
    timeframe="1h",
    direction=TradeDirection.SHORT,  # Trade bajista
    confidence=0.85,
    key_levels={
        "peak1": 52000,
        "peak2": 51800,
        "neckline": 50500,
        "target": 49000,
    },
)

# Verificar dirección
if pattern.direction == TradeDirection.SHORT:
    print("Patrón bajista - abrir posición SHORT")
elif pattern.direction == TradeDirection.LONG:
    print("Patrón alcista - abrir posición LONG")
```

---

## Crear un Nuevo Patrón

### Paso 1: Crear Archivo

Crear `app/patterns/reversal/mi_patrón.py` o `app/patterns/continuation/mi_patrón.py`.

### Paso 2: Implementar Clase

```python
from __future__ import annotations
from typing import Optional
import numpy as np
from app.market.candles.models import Candle
from app.patterns.base_pattern import BasePattern, PatternResult, PatternType, TradeDirection
from app.patterns.registry import register_pattern

@register_pattern
class MiPatron(BasePattern):
    @property
    def name(self) -> str:
        return "mi_patron"
    
    @property
    def pattern_type(self) -> PatternType:
        return PatternType.REVERSAL  # o CONTINUATION
    
    @property
    def max_confirmation_candles(self) -> int:
        return 20  # Configurar según necesidad
    
    def detect(
        self,
        candles: list[Candle],
        symbol: str,
        timeframe: str,
    ) -> Optional[PatternResult]:
        """Detectar el patrón."""
        if len(candles) < 10:
            return None
        
        # Tu lógica de detección aquí
        highs = np.array([c.data.high for c in candles])
        lows = np.array([c.data.low for c in candles])
        
        # Ejemplo: detectar dos mínimos similares
        # Usar scipy.signal.find_peaks (implementado en C, ~50-100x más rápido)
        from scipy.signal import find_peaks
        troughs, _ = find_peaks(-lows, distance=3)
        if len(troughs) < 2:
            return None
        
        # Verificar condiciones
        trough1 = lows[troughs[0]]
        trough2 = lows[troughs[1]]
        
        if abs(trough1 - trough2) / trough1 > 0.02:
            return None
        
        # Crear resultado
        neckline = np.max(highs[troughs[0]:troughs[1]])
        pattern_height = neckline - trough1
        
        return PatternResult(
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            symbol=symbol,
            timeframe=timeframe,
            direction=TradeDirection.LONG,  # Patrón alcista
            confidence=0.85,
            key_levels={
                "trough1": trough1,
                "trough2": trough2,
                "neckline": neckline,
                "target": neckline + pattern_height,
            },
            max_confirmation_candles=self.max_confirmation_candles,
        )
    
    def validate(self, pattern: PatternResult, candles: list[Candle]) -> bool:
        """Validar si el patrón sigue activo."""
        if not pattern.key_levels:
            return False
        
        neckline = pattern.key_levels.get("neckline", 0)
        if not candles:
            return False
        
        latest_close = candles[-1].data.close
        return latest_close > neckline * 0.98
    
    def _find_troughs(self, data: np.ndarray, distance: int = 3) -> list[int]:
        """Encontrar mínimos locales usando scipy (C-level)."""
        from scipy.signal import find_peaks
        idx, _ = find_peaks(-data, distance=distance)
        return idx.tolist()
```

> **Nota**: `BasePattern` ya no define `score()`. La puntuación la calcula el `ScoringEngine` (pesos en YAML + componente ML de conocimiento), no cada patrón.

### Paso 3: Registrar

El decorador `@register_pattern` automáticamente registra el patrón.

### Paso 4: Probar

```python
from app.patterns.registry import PatternRegistry

# Verificar que se registró
patterns = PatternRegistry.get_all()
assert "mi_patron" in patterns

# Probar detección
detector = PatternRegistry.get("mi_patron")()
result = detector.detect(candles, "BTCUSDT", "1h")
```

---

## Ciclo de Vida de un Patrón

```
DETECTED → FORMING → WAITING_BREAKOUT → CONFIRMED → SIGNAL_SENT → OPEN → CLOSED
                                                     ↓
                                               INVALIDATED / EXPIRED / REJECTED
```

### Estados Explicados

| Estado | Descripción |
|--------|-------------|
| `DETECTED` | Patrón detectado inicialmente |
| `FORMING` | Estructura formándose |
| `WAITING_BREAKOUT` | Esperando ruptura de nivel clave |
| `CONFIRMED` | Ruptura confirmada |
| `SIGNAL_SENT` | Señal enviada al usuario |
| `OPEN` | Operación abierta |
| `TP_HIT` | Take Profit alcanzado |
| `SL_HIT` | Stop Loss alcanzado |
| `CLOSED` | Operación cerrada |
| `INVALIDATED` | Patrón invalidado (deformación) |
| `EXPIRED` | Tiempo máximo de confirmación |
| `CANCELLED` | Cancelado por el usuario |
| `REJECTED` | Rechazado por baja calidad |

Cada transición queda registrada con `from_state`, `to_state`, `timestamp` y
`reason` (`LifecycleTransition` en `app/lifecycle/models.py`).

### Motor de Ciclo de Vida

El `LifecycleEngine` (`app/lifecycle/engine.py`) es un motor **independiente** de
los detectores. Los detectores **solo detectan estructura** (`BasePattern.detect`),
el motor administra los estados:

```python
from app.lifecycle.engine import LifecycleEngine
from app.lifecycle.models import LifecycleState

engine = LifecycleEngine()

# Registrar un patrón detectado (estado inicial: DETECTED)
lifecycle = await engine.register(pattern)

# Transicionar estado
await engine.transition(
    lifecycle.id,
    LifecycleState.WAITING_BREAKOUT,
    reason="Niveles clave definidos",
)

# Sincronizar desde el estado del patrón
await engine.update_pattern_status(pattern, PatternStatus.CONFIRMED)
```

### Vida Útil / EXPIRED

Cada patrón define su `max_confirmation_candles`. Si el precio no rompe en ese
tiempo, el patrón pasa a `EXPIRED`:

```python
class DoubleBottomPattern(BasePattern):
    def max_confirmation_candles(self) -> int:
        return 20  # Se agota en 20 velas

class BullFlagPattern(BasePattern):
    def max_confirmation_candles(self) -> int:
        return 12
```

`BasePattern.update()` incrementa el contador de velas y transiciona a `EXPIRED`
cuando se alcanza el máximo.

---

## Health Score

Cada patrón tiene una **salud dinámica (0-100)** recalculada con **cada vela** por
el `HealthEngine` (`app/health/engine.py`):

| Factor | Peso | Qué evalúa |
|--------|------|------------|
| `time_decay` | 20% | Velas restantes antes de expirar |
| `deformation` | 20% | Integridad estructural (`validate()`) |
| `volume` | 15% | El volumen confirma el movimiento |
| `trend` | 10% | Alineación con la tendencia principal |
| `atr` | 10% | Volatilidad ATR suficiente |
| `slope` | 10% | Pendiente reciente del precio |
| `false_breakouts` | 10% | Rupturas falsas del nivel clave |
| `volatility` | 5% | Consistencia de la volatilidad |

```python
from app.health.engine import HealthEngine

engine = HealthEngine()
report = await engine.calculate(pattern, detector, candles, indicators)

print(report.health)          # 0-100
print(report.weakest_factor)  # Factor que más daña la salud
pattern.update_health(report.health)
```

**Ejemplo de evolución**:

```
Vela 1:  Health ██████████ 100
Vela 5:  Health ████████   82
Vela 10: Health ██████     65
Vela 15: Health ████       43
Vela 20: Health ██         21
Vela 21: Estado → EXPIRED
```

---

## Sistema de Confirmación

Detectar un patrón **NO implica enviar una señal**. Antes debe pasar por el
`ConfirmationEngine` (`app/confirmation/engine.py`):

| Regla | Requerida | Qué valida |
|-------|-----------|------------|
| `breakout` | ✅ | Ruptura del nivel clave (`neckline`) |
| `volume_confirmation` | ✅ | Volumen confirma la ruptura |
| `risk_reward` | ✅ | R/R ≥ 2.0 (riesgo mínimo) |
| `atr_sufficient` | | ATR suficiente para el movimiento |
| `trend_alignment` | | Alineación con la tendencia |
| `liquidity` | | Liquidez sostenida (CV de volumen) |
| `spread_acceptable` | | Spread aceptable |
| `distance_to_support` | | Distancia al soporte/resistencia razonable |

Una señal solo se genera si todas las reglas **requeridas** pasan y el score de
confirmación ≥ 60.

---

## Pipeline de Patrones

El `PatternPipeline` (`app/patterns/pipeline.py`) **orquesta el flujo completo**
por cada símbolo/timeframe:

```
Datos → Indicadores → Detección → Lifecycle → Health → Confirmación → Scoring
     → Hipótesis → Estrategia → Señal → Telegram
```

El pipeline **no genera señales directamente desde el patrón**: tras el scoring
emite una **hipótesis** (`PatternHypothesis`: `PatternResult` + indicadores +
score + health + confirmación) y el `StrategyEngine` la evalúa con las
estrategias habilitadas. Solo si alguna estrategia decide **ENTER** se llama a
`SignalEngine.create_signal(..., strategy_signal=...)` y se envía por Telegram
si el score ≥ 95. Las decisiones de cada estrategia quedan en
`result.metadata["strategy_decisions"]`.

```python
from app.patterns.pipeline import PatternPipeline

pipeline = PatternPipeline()  # usa el provider configurado

# Procesar un símbolo (obtiene velas y corre todo el flujo)
stats = await pipeline.process_symbol("BTCUSDT", "1h")
print(stats)  # tracked / active / expired / confirmed / signals_sent
```

`PatternService` (`app/patterns/service.py`) ejecuta el pipeline de forma
periódica (vía `Scheduler`) y se arranca automáticamente con la API en el
lifespan de FastAPI.

### Estrategias

Las estrategias viven en `app/strategy/` y se registran con `@register_strategy`.
Estrategias incluidas:

| Estrategia | Tipo | Lógica |
|------------|------|--------|
| `trend_follow` | Continuación | Entra a favor de la tendencia (EMA9 > EMA21 en LONG, momentum positivo) |
| `breakout` | Ruptura | Entra con momentum direccional y RSI en rango medio |
| `contrarian` | Reversa | Entra en reversales (RSI extremo + momentum perdiendo fuerza), solo patrones `REVERSAL` |

Se habilitan y parametrizan en `config/settings.yaml` (sección `strategies:`),
ver `docs/CONFIGURATION.md`. Las estrategias pueden compararse sobre las mismas
detecciones con `compare_strategies`/`run_strategy_backtest`
(`app/strategy/evaluator.py`).

---

## Scoring System

El score combina múltiples factores:

| Factor | Peso | Descripción |
|--------|------|-------------|
| Pattern Structure | 35% | Calidad estructural del patrón |
| Volume | 20% | Volumen de confirmación |
| Momentum | 10% | RSI, MACD |
| ATR | 10% | Volatilidad adecuada |
| RSI | 10% | Sobrecompra/venta |
| MACD | 5% | Señales de momentum |
| EMA | 5% | Alineación de tendencia |
| ML History | 5% | Predicción del modelo |

**Interpretación del Score**:

| Score | Acción |
|-------|--------|
| < 60 | Ignorar |
| 60-75 | Observar |
| 75-85 | Preparar |
| 85-95 | Alta prioridad |
| 95+ | Enviar Telegram |

---

## Ejemplo Completo

```python
import asyncio
from app.data.providers import DataProviderFactory
from app.market.candles.models import Candle, CandleData
from app.market.indicators.calculator import IndicatorCalculator
from app.patterns.registry import PatternRegistry
from app.scoring.engine import ScoringEngine
from app.lifecycle.engine import LifecycleEngine

async def analyze_market():
    # 1. Conectar al proveedor (binance, bybit, yahoo, polygon,
    #    alphavantage, metatrader, interactive_brokers)
    provider = DataProviderFactory.create("binance")
    await provider.connect()
    
    # 2. Obtener datos
    candles_data = await provider.get_history(
        symbol="BTCUSDT",
        timeframe="1h",
        limit=200
    )
    
    # 3. Convertir a modelos internos
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
    
    # 4. Calcular indicadores
    indicator_calc = IndicatorCalculator()
    indicators = indicator_calc.get_latest_indicators(candles)
    
    # 5. Detectar patrones
    detectors = PatternRegistry.get_all_instances()
    lifecycle_engine = LifecycleEngine()
    scoring_engine = ScoringEngine()
    
    for detector in detectors:
        result = detector.detect(candles, "BTCUSDT", "1h")
        
        if result:
            print(f"\n{'='*50}")
            print(f"Patrón detectado: {result.pattern_name}")
            print(f"Dirección: {result.direction.value}")  # LONG o SHORT
            print(f"Confianza: {result.confidence:.2%}")
            
            # Registrar en lifecycle
            lifecycle = await lifecycle_engine.register(result)
            
            # Calcular score
            score = scoring_engine.calculate_score(result, indicators, candles)
            print(f"Score: {score.total_score:.1f}/100 ({score.grade})")
            
            # Actualizar patrón con score
            result.score = score.total_score
            
            print(f"Niveles clave:")
            for level, price in result.key_levels.items():
                print(f"  {level}: ${price:,.2f}")
    
    # 6. Ver estadísticas
    stats = lifecycle_engine.get_statistics()
    print(f"\nEstadísticas del Lifecycle:")
    for state, count in stats.items():
        print(f"  {state}: {count}")
    
    await provider.disconnect()

# Ejecutar
asyncio.run(analyze_market())
```

---

## Mejores Prácticas

1. **Mínimo de velas**: Siempre verificar que hay suficientes velas antes de detectar
2. **Timeframes múltiples**: Confirmar patrones en múltiples timeframes
3. **Volumen**: Un patrón sin volumen de confirmación es menos confiable
4. **Contexto**: Considerar la tendencia general del mercado
5. **Gestión de riesgo**: Nunca arriesgar más del 2% por operación
6. **Backtesting**: Probar cualquier nuevo patrón antes de usarlo en real
