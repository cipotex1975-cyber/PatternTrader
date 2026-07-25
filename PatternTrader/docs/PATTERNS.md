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
from app.patterns.reversal.double_top import DoubleTopPattern
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

## Interfaz de Patrones

Todos los patrones heredan de `BasePattern`:

```python
from app.patterns.base_pattern import BasePattern, PatternResult, PatternType

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
    
    @abstractmethod
    def score(self, pattern, indicators) -> float:
        """Calcular score del patrón."""
        ...
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
from app.patterns.base_pattern import BasePattern, PatternResult, PatternType
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
        troughs = self._find_troughs(lows)
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
    
    def score(self, pattern: PatternResult, indicators: dict[str, float]) -> float:
        """Calcular score basado en indicadores."""
        score = 50.0
        
        rsi = indicators.get("rsi", 50)
        if rsi < 30:  # Sobreventa
            score += 15
        elif rsi < 40:
            score += 10
        
        macd = indicators.get("macd", 0)
        macd_signal = indicators.get("macd_signal", 0)
        if macd > macd_signal:  # Cruce alcista
            score += 10
        
        return min(100.0, score)
    
    def _find_troughs(self, data: np.ndarray, distance: int = 3) -> list[int]:
        """Encontrar mínimos locales."""
        troughs = []
        for i in range(distance, len(data) - distance):
            if all(data[i] <= data[i - j] for j in range(1, distance + 1)) and \
               all(data[i] <= data[i + j] for j in range(1, distance + 1)):
                troughs.append(i)
        return troughs
```

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
                                              INVALIDATED / EXPIRED
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
| `INVALIDATED` | Patrón invalidado |
| `EXPIRED` | Tiempo máximo de confirmación |
| `CANCELLED` | Cancelado por el usuario |
| `REJECTED` | Rechazado por baja calidad |

---

## Health Score

Cada patrón tiene un **Health Score** (0-100) que se recalcula con cada vela:

```python
# Factores que afectan el Health Score
health_factors = {
    "tiempo_transcurrido": -0.5,  # Por cada vela
    "volatilidad": +0.3,          # Si ATR es favorable
    "volumen": +0.2,              # Si volumen confirma
    "deformacion": -0.4,          # Si el patrón se deforma
    "rupturas_falsas": -0.5,     # Por cada ruptura falsa
}
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
| 95+ | Enviar señal |

---

## Ejemplo Completo

```python
import asyncio
from app.data.providers.binance import BinanceProvider
from app.market.candles.models import Candle, CandleData
from app.market.indicators.calculator import IndicatorCalculator
from app.patterns.registry import PatternRegistry
from app.scoring.engine import ScoringEngine
from app.lifecycle.engine import LifecycleEngine

async def analyze_market():
    # 1. Conectar al proveedor
    provider = BinanceProvider()
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
