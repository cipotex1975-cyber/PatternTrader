# Guía de Backtesting

## Visión General

PatternTrader incluye un motor de backtesting completo para evaluar estrategias de trading basadas en patrones chartistas. El motor simula operaciones sobre datos históricos y calcula métricas profesionales.

---

## Arquitectura del Backtesting

```
┌─────────────────────────────────────────────────────────────┐
│                    BacktestEngine                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │    Data     │  │   Pattern   │  │    Trade            │ │
│  │   Feed      │  │  Detector   │  │    Executor         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Risk      │  │  Position   │  │    Metrics          │ │
│  │   Manager   │  │   Sizer     │  │    Calculator       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Uso Básico

### Script de Backtesting con Datos Reales

El proyecto incluye un script completo `run_backtest.py` que carga datos históricos, detecta patrones automáticamente y ejecuta el backtest. Cada par de divisas tiene su propia configuración en `config/pairs.yaml`.

```bash
# Ejecutar backtest para un par (usa configuración del par)
python run_backtest.py --pair USDCAD
python run_backtest.py --pair USDJPY

# Sobreescribir exclude desde CLI
python run_backtest.py --pair USDJPY --exclude "head_and_shoulders"

# Usar archivo de datos específico
python run_backtest.py --pair USDCAD --data /ruta/a/misdatos.txt

# Limitar a N velas
python run_backtest.py --pair USDCAD --max-candles 20000

# Ejecución rápida
python run_backtest.py --pair USDCAD --step 200 --max-patterns 300
```

**Parámetros CLI** de `run_backtest.py`:

| Parámetro | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| `--pair` | `None` (USDCAD) | Par de divisas para usar configuración (ej: USDCAD, USDJPY) |
| `--data` | auto (según par) | Ruta al archivo de datos |
| `--max-candles` | `None` (todas) | Número máximo de velas a usar |
| `--step` | según par | Paso entre ventanas deslizantes |
| `--max-patterns` | según par | Número máximo de patrones a detectar |
| `--exclude` | según par | Patrones a excluir, separados por coma |

**Patrones disponibles**: `double_bottom`, `double_top`, `inverse_head_and_shoulders`, `head_and_shoulders`, `bull_flag`, `bear_flag`, `bull_pennant`, `bear_pennant`

### Configuración por Par

Cada par tiene sus propios parámetros en `config/pairs.yaml`:

```yaml
default:                    # Valores por defecto para todos los pares
  window: 200
  step: 100
  sl_tp:
    double_top:
      sl_method: peaks_midpoint
      sl_buffer: 0.002

pairs:
  USDCAD:                   # Configuración específica de USDCAD
    timeframe: H1
    window: 200
    step: 100
    exclude: []

  USDJPY:                   # Configuración específica de USDJPY
    timeframe: H1
    window: 150             # Ventana más corta
    step: 75
    sl_tp:
      double_top:
        sl_method: neckline
        sl_buffer: 0.004    # Stop más amplio
    exclude: []
```

**SL methods disponibles:**

| Método | Descripción | Ejemplo |
|--------|-------------|---------|
| `peaks` | Encima de los peaks (original) | `max(peak1, peak2) * (1 + buffer)` |
| `peaks_midpoint` | Punto medio entre neckline y peaks | `(neckline + max(peak1, peak2)) / 2 * (1 + buffer)` |
| `neckline` | Encima del neckline (tight) | `neckline * (1 + buffer)` |
| `troughs` | Debajo de los troughs | `min(trough1, trough2) * (1 - buffer)` |
| `flag_low` | Debajo del最低点 del flag | `flag_low * (1 - buffer)` |
| `flag_high` | Encima del最高点 del flag | `flag_high * (1 + buffer)` |

**Cómo elegir la configuración:**

1. **Mercado en tendencia**: Excluir patrones SHORT (`--exclude bear_flag,double_top`)
2. **Mercado volátil**: Usar `sl_buffer` más amplio (0.004-0.005)
3. **Mercado lateral**: Usar `sl_buffer` más ajustado (0.001-0.002)
4. **Más señales**: Reducir `window` y `step`
5. **Mejor calidad**: Aumentar `window` y usar `--exclude` para patrones débiles

### ¿Qué es una ventana deslizante?

El detector de patrones usa una **ventana deslizante** para escanear todo el histórico de velas. Cada ventana es un bloque连续 de velas que se analiza para detectar formaciones chartistas:

```
Velas: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...]

Con window=200, step=100:

Ventana 1: [vela 1 ... vela 200]    ← analiza estas 200 velas
Ventana 2: [vela 101 ... vela 300]  ← se mueve 100 velas hacia adelante
Ventana 3: [vela 201 ... vela 400]  ← se mueve 100 más
...
```

- **`window`**: Cuántas velas analiza cada detector (mínimo necesario para formar un patrón)
- **`step`**: Cuánto se mueve la ventana entre cada iteración

Con 50,000 velas y `step=100`, se analizan ~500 ventanas × 8 detectores = ~4,000 llamadas de detección.

**Pipeline del script:**

1. **Carga de datos**: Lee archivos tab-delimited (formato MT4/MT5)
2. **Detección de patrones**: Ventana deslizante con todos los detectores registrados
3. **Deduplicación**: Evita patrones duplicados por ventana
4. **Preparación SL/TP**: Calcula stop loss y take profit según el patrón
5. **Ejecución del backtest**: Motor de trading con gestión de posiciones
6. **Análisis de resultados**: Métricas por patrón, mejores/peores trades

### Ejemplo Simple

```python
import asyncio
from app.backtesting.engine import BacktestEngine
from app.backtesting.models import BacktestConfig
from app.market.candles.models import Candle, CandleData
from app.patterns.base_pattern import PatternResult, PatternType, PatternStatus, TradeDirection
from datetime import datetime, timezone, timedelta
import random

async def simple_backtest():
    # 1. Generar datos de ejemplo
    candles = []
    base_price = 50000
    
    for i in range(500):
        change = random.uniform(-500, 500)
        open_price = base_price + change
        high = open_price + random.uniform(0, 300)
        low = open_price - random.uniform(0, 300)
        close = open_price + random.uniform(-200, 200)
        
        candles.append(Candle(
            symbol="BTCUSDT",
            timeframe="1h",
            data=CandleData(
                timestamp=datetime.now(timezone.utc) - timedelta(hours=500-i),
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=random.randint(1000, 10000),
            )
        ))
        base_price = close
    
    # 2. Crear patrones de ejemplo
    patterns = [
        PatternResult(
            pattern_name="double_bottom",
            pattern_type=PatternType.REVERSAL,
            symbol="BTCUSDT",
            timeframe="1h",
            direction=TradeDirection.LONG,
            confidence=0.8,
            status=PatternStatus.CONFIRMED,
            entry_price=50500,
            stop_loss=49500,
            take_profit=53000,
        ),
        PatternResult(
            pattern_name="bull_flag",
            pattern_type=PatternType.CONTINUATION,
            symbol="BTCUSDT",
            timeframe="1h",
            direction=TradeDirection.LONG,
            confidence=0.85,
            status=PatternStatus.CONFIRMED,
            entry_price=51000,
            stop_loss=50200,
            take_profit=53500,
        ),
    ]
    
    # 3. Configurar backtest
    config = BacktestConfig(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005,
        max_positions=5,
        risk_per_trade=0.02,
    )
    
    # 4. Ejecutar backtest
    engine = BacktestEngine(config)
    result = engine.run(candles, patterns)
    
    # 5. Mostrar resultados
    print("=" * 60)
    print("RESULTADOS DEL BACKTEST")
    print("=" * 60)
    print(f"Período: {result.start_date.date()} a {result.end_date.date()}")
    print(f"Capital inicial: ${result.initial_capital:,.2f}")
    print(f"Capital final: ${result.final_capital:,.2f}")
    print(f"Retorno total: {result.total_return:.2%}")
    print()
    print("MÉTRICAS:")
    print(f"  Trades totales: {result.metrics.total_trades}")
    print(f"  Trades ganadores: {result.metrics.winning_trades}")
    print(f"  Trades perdedores: {result.metrics.losing_trades}")
    print(f"  Win Rate: {result.metrics.win_rate:.2%}")
    print(f"  Profit Factor: {result.metrics.profit_factor:.2f}")
    print(f"  Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {result.metrics.max_drawdown_pct:.2f}%")
    print(f"  Expectancy: ${result.metrics.expectancy:,.2f}")
    print()

asyncio.run(simple_backtest())
```

---

## Configuración

### BacktestConfig

```python
from app.backtesting.models import BacktestConfig

config = BacktestConfig(
    # Capital inicial
    initial_capital=100000,
    
    # Comisión por trade (0.1%)
    commission=0.001,
    
    # Slippage estimado (0.05%)
    slippage=0.0005,
    
    # Máximo de posiciones abiertas
    max_positions=10,
    
    # Riesgo por trade (2% del capital)
    risk_per_trade=0.02,
    
    # Pérdida diaria máxima (6% del capital)
    max_daily_loss=0.06,
    
    # Trailing stop
    use_trailing_stop=False,
    trailing_stop_pct=0.02,
)
```

### Parámetros Explicados

| Parámetro | Descripción | Recomendado |
|-----------|-------------|-------------|
| initial_capital | Capital de inicio | Según tu inversión real |
| commission | Comisión por trade | 0.001 (0.1%) |
| slippage | Diferencia estimada | 0.0005 (0.05%) |
| max_positions | Posiciones simultáneas | 5-10 |
| risk_per_trade | Riesgo por operación | 0.01-0.02 |
| max_daily_loss | Pérdida diaria máxima | 0.04-0.06 |

---

## Métricas de Resultado

### Métricas Principales

```python
# Win Rate
win_rate = result.metrics.win_rate
# Ejemplo: 0.65 = 65% de trades ganadores

# Profit Factor
profit_factor = result.metrics.profit_factor
# Ejemplo: 2.0 = Ganas el doble de lo que pierdes

# Sharpe Ratio
sharpe = result.metrics.sharpe_ratio
# Ejemplo: 1.5 = Buen ajuste riesgo/retorno

# Max Drawdown
max_dd = result.metrics.max_drawdown_pct
# Ejemplo: 15% = Pérdida máxima desde un máximo
```

### Todas las Métricas

```python
metrics = {
    # Trading
    "total_trades": 45,           # Total de operaciones
    "winning_trades": 28,         # Trades ganadores
    "losing_trades": 17,          # Trades perdedores
    "win_rate": 0.622,            # Porcentaje de ganancia
    
    # Ganancias
    "average_win": 850.0,         # Ganancia promedio
    "average_loss": -420.0,       # Pérdida promedio
    "expectancy": 277.8,          # Expectativa por trade
    
    # Riesgo/Retorno
    "profit_factor": 2.1,         # Factor de beneficio
    "sharpe_ratio": 1.8,          # Ratio de Sharpe
    "sortino_ratio": 2.2,         # Ratio de Sortino
    "calmar_ratio": 1.2,          # Ratio de Calmar
    
    # Drawdown
    "max_drawdown": 5200.0,       # Drawdown máximo ($)
    "max_drawdown_pct": 5.2,      # Drawdown máximo (%)
    
    # Totales
    "total_pnl": 12500.0,         # PnL total
    "total_pnl_pct": 12.5,        # Retorno total (%)
    "annual_return": 0.25,        # Retorno anualizado
    "volatility": 0.15,           # Volatilidad
}
```

---

## Análisis de Trades

### Listar Trades

```python
# Todas las trades
for trade in result.trades:
    print(f"{trade.symbol} | {trade.direction} | "
          f"Entry: ${trade.entry_price:,.2f} | "
          f"Exit: ${trade.exit_price:,.2f} | "
          f"PnL: ${trade.pnl:,.2f} | "
          f"Status: {trade.status}")
```

### Filtrar Trades

```python
# Trades ganadores
winning_trades = [t for t in result.trades if t.pnl > 0]

# Trades perdedores
losing_trades = [t for t in result.trades if t.pnl <= 0]

# Trades por símbolo
btc_trades = [t for t in result.trades if t.symbol == "BTCUSDT"]

# Trades por patrón
double_top_trades = [t for t in result.trades if t.pattern_name == "double_top"]
```

### Estadísticas por Patrón

```python
from collections import defaultdict

def analyze_by_pattern(trades):
    stats = defaultdict(lambda: {"count": 0, "wins": 0, "total_pnl": 0})
    
    for trade in trades:
        pattern = trade.pattern_name
        stats[pattern]["count"] += 1
        stats[pattern]["total_pnl"] += trade.pnl
        if trade.pnl > 0:
            stats[pattern]["wins"] += 1
    
    for pattern, data in stats.items():
        win_rate = data["wins"] / data["count"] if data["count"] > 0 else 0
        print(f"{pattern}:")
        print(f"  Trades: {data['count']}")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  PnL Total: ${data['total_pnl']:,.2f}")

analyze_by_pattern(result.trades)
```

---

## Curva de Capital

```python
# Obtener curva de capital
equity_curve = result.equity_curve

# Graficar con Plotly
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[e["timestamp"] for e in equity_curve],
    y=[e["equity"] for e in equity_curve],
    mode="lines",
    name="Equity"
))

fig.update_layout(
    title="Curva de Capital",
    xaxis_title="Tiempo",
    yaxis_title="Capital ($)",
    template="plotly_dark"
)

fig.show()
```

---

## Backtesting Múltiple

### Probar Múltiples Configuraciones

```python
async def optimize_backtest():
    """Probar diferentes configuraciones."""
    
    configs = [
        BacktestConfig(risk_per_trade=0.01, max_positions=5),
        BacktestConfig(risk_per_trade=0.02, max_positions=10),
        BacktestConfig(risk_per_trade=0.03, max_positions=15),
    ]
    
    results = []
    for config in configs:
        engine = BacktestEngine(config)
        result = engine.run(candles, patterns)
        results.append({
            "config": config,
            "return": result.total_return,
            "sharpe": result.metrics.sharpe_ratio,
            "max_dd": result.metrics.max_drawdown_pct,
        })
    
    # Comparar resultados
    for i, res in enumerate(results):
        print(f"Config {i+1}:")
        print(f"  Return: {res['return']:.2%}")
        print(f"  Sharpe: {res['sharpe']:.2f}")
        print(f"  Max DD: {res['max_dd']:.2f}%")
```

---

## Walk Forward Analysis

```python
async def walk_forward_analysis(candles, patterns, n_splits=5):
    """Análisis walk-forward."""
    
    split_size = len(candles) // n_splits
    results = []
    
    for i in range(n_splits):
        # Dividir datos
        train_end = split_size * (i + 1)
        test_start = train_end
        test_end = min(test_start + split_size, len(candles))
        
        train_candles = candles[:train_end]
        test_candles = candles[test_start:test_end]
        
        # Entrenar en datos de entrenamiento
        # (En este caso, usaríamos los mismos patrones)
        
        # Evaluar en datos de test
        engine = BacktestEngine()
        result = engine.run(test_candles, patterns)
        
        results.append({
            "split": i + 1,
            "return": result.total_return,
            "win_rate": result.metrics.win_rate,
            "sharpe": result.metrics.sharpe_ratio,
        })
    
    # Promediar resultados
    avg_return = sum(r["return"] for r in results) / len(results)
    avg_sharpe = sum(r["sharpe"] for r in results) / len(results)
    
    print(f"Walk-Forward Results:")
    print(f"  Average Return: {avg_return:.2%}")
    print(f"  Average Sharpe: {avg_sharpe:.2f}")
    
    return results
```

---

## Monte Carlo Simulation

```python
import numpy as np

def monte_carlo_simulation(trades, n_simulations=1000):
    """Simulación Monte Carlo."""
    
    pnls = [t.pnl for t in trades]
    
    results = []
    for _ in range(n_simulations):
        # Mezclar trades aleatoriamente
        shuffled = np.random.permutation(pnls)
        
        # Simular equity curve
        equity = [100000]
        for pnl in shuffled:
            equity.append(equity[-1] + pnl)
        
        results.append({
            "final_equity": equity[-1],
            "max_drawdown": max(0, max(np.maximum.accumulate(equity)) - equity[-1]),
        })
    
    # Estadísticas
    final_equities = [r["final_equity"] for r in results]
    max_dds = [r["max_drawdown"] for r in results]
    
    print(f"Monte Carlo Results ({n_simulations} simulations):")
    print(f"  Median Final Equity: ${np.median(final_equities):,.2f}")
    print(f"  5th Percentile: ${np.percentile(final_equities, 5):,.2f}")
    print(f"  95th Percentile: ${np.percentile(final_equities, 95):,.2f}")
    print(f"  Probability of Profit: {sum(1 for e in final_equities if e > 100000) / n_simulations:.2%}")
    
    return results
```

---

## Ejemplo Completo

```python
import asyncio
from app.backtesting.engine import BacktestEngine
from app.backtesting.models import BacktestConfig, BacktestResult
from app.market.candles.models import Candle, CandleData
from app.patterns.base_pattern import PatternResult, PatternType, PatternStatus, TradeDirection
from datetime import datetime, timezone, timedelta
import random

async def complete_backtest_analysis():
    """Análisis completo de backtesting."""
    
    # 1. Generar datos realistas
    print("Generando datos...")
    candles = generate_realistic_candles(1000)
    
    # 2. Detectar patrones
    print("Detectando patrones...")
    patterns = detect_patterns_from_candles(candles)
    
    # 3. Configurar backtest
    config = BacktestConfig(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005,
        max_positions=10,
        risk_per_trade=0.02,
    )
    
    # 4. Ejecutar backtest
    print("Ejecutando backtest...")
    engine = BacktestEngine(config)
    result = engine.run(candles, patterns)
    
    # 5. Analizar resultados
    print("\n" + "=" * 60)
    print("ANÁLISIS COMPLETO")
    print("=" * 60)
    
    # Métricas principales
    print(f"\nRESUMEN:")
    print(f"  Capital Inicial: ${result.initial_capital:,.2f}")
    print(f"  Capital Final: ${result.final_capital:,.2f}")
    print(f"  Retorno Total: {result.total_return:.2%}")
    print(f"  Retorno Anualizado: {result.annualized_return:.2%}")
    
    print(f"\nMÉTRICAS DE TRADING:")
    print(f"  Total Trades: {result.metrics.total_trades}")
    print(f"  Win Rate: {result.metrics.win_rate:.2%}")
    print(f"  Profit Factor: {result.metrics.profit_factor:.2f}")
    print(f"  Expectancy: ${result.metrics.expectancy:,.2f}")
    
    print(f"\nMÉTRICAS DE RIESGO:")
    print(f"  Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {result.metrics.sortino_ratio:.2f}")
    print(f"  Max Drawdown: {result.metrics.max_drawdown_pct:.2f}%")
    print(f"  Volatility: {result.metrics.volatility:.2%}")
    
    # Análisis por patrón
    print(f"\nANÁLISIS POR PATRÓN:")
    analyze_by_pattern(result.trades)
    
    # Análisis por dirección
    print(f"\nANÁLISIS POR DIRECCIÓN:")
    long_trades = [t for t in result.trades if t.direction.value == "LONG"]
    short_trades = [t for t in result.trades if t.direction.value == "SHORT"]
    
    long_wr = sum(1 for t in long_trades if t.pnl > 0) / len(long_trades) if long_trades else 0
    short_wr = sum(1 for t in short_trades if t.pnl > 0) / len(short_trades) if short_trades else 0
    
    print(f"  Long Trades: {len(long_trades)} | Win Rate: {long_wr:.2%}")
    print(f"  Short Trades: {len(short_trades)} | Win Rate: {short_wr:.2%}")
    
    # Mejores y peores trades
    print(f"\nMEJORES TRADES:")
    best_trades = sorted(result.trades, key=lambda t: t.pnl, reverse=True)[:5]
    for t in best_trades:
        print(f"  {t.symbol} | {t.pattern_name} | PnL: ${t.pnl:,.2f}")
    
    print(f"\nPEORES TRADES:")
    worst_trades = sorted(result.trades, key=lambda t: t.pnl)[:5]
    for t in worst_trades:
        print(f"  {t.symbol} | {t.pattern_name} | PnL: ${t.pnl:,.2f}")
    
    return result

def generate_realistic_candles(n: int) -> list[Candle]:
    """Generar velas realistas."""
    candles = []
    base_price = 50000
    trend = 0.001  # Tendencia alcista
    
    for i in range(n):
        # Movimiento con tendencia y ruido
        change = random.gauss(trend, 0.02) * base_price
        open_price = base_price
        close_price = open_price + change
        
        # High y Low
        volatility = abs(change) * random.uniform(0.5, 1.5)
        high = max(open_price, close_price) + random.uniform(0, volatility)
        low = min(open_price, close_price) - random.uniform(0, volatility)
        
        # Volumen
        volume = random.randint(5000, 50000)
        
        candles.append(Candle(
            symbol="BTCUSDT",
            timeframe="1h",
            data=CandleData(
                timestamp=datetime.now(timezone.utc) - timedelta(hours=n-i),
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=volume,
            )
        ))
        
        base_price = close_price
    
    return candles

asyncio.run(complete_backtest_analysis())
```

---

## Motor Independiente de Backtesting

Además del `BacktestEngine` clásico, la plataforma incluye un conjunto de módulos independientes para ejecutar y validar estrategias de forma profesional: `BacktestRunner`, validaciones temporales, simulaciones Monte Carlo y optimización de parámetros.

```
┌────────────────────────────────────────────────────────────────────┐
│                       BacktestRunner                                │
│  simple · múltiple · paralelo (ThreadPool)                          │
├────────────────────────────────────────────────────────────────────┤
│  Validación                Optimización              Métricas      │
│  ┌───────────────────┐   ┌───────────────────┐   ┌──────────────┐  │
│  │ Walk Forward      │   │ Grid Search       │   │ Sharpe       │  │
│  │ Out of Sample     │   │ Random Search     │   │ Sortino      │  │
│  │ Rolling Window    │   │ Bayesiana (Optuna)│   │ Calmar       │  │
│  │ Cross Validation  │   └───────────────────┘   │ Ulcer Index  │  │
│  │ Monte Carlo       │                           │ Expectancy   │  │
│  └───────────────────┘                           │ Precision…   │  │
│                                                   └──────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

### Módulos

| Módulo | Clase | Descripción |
|--------|-------|-------------|
| `app/backtesting/runner.py` | `BacktestRunner` | Backtest simple, múltiple y paralelo |
| `app/backtesting/validation.py` | `WalkForwardValidator` | Validación walk-forward anclada |
| | `OutOfSampleValidator` | División entrenamiento/test |
| | `RollingWindowValidator` | Ventanas rodantes de tamaño fijo |
| | `CrossValidator` | Validación cruzada por bloques temporales |
| | `MonteCarloSimulator` | Re-muestreo de trades y percentiles |
| | `TimeSeriesSplitter` | Splits temporales reutilizables |
| `app/backtesting/optimization.py` | `BacktestOptimizer` | Grid, random y bayesiana (Optuna) |
| `app/backtesting/metrics.py` | `MetricsCalculator` | Métricas profesionales + clasificación |

### Backtest Simple y Múltiple

```python
from app.backtesting.runner import BacktestRunner
from app.backtesting.models import BacktestConfig

runner = BacktestRunner()

# Simple
result = runner.run(candles, patterns)

# Múltiple: probar varias configuraciones
results = runner.run_multiple([
    {"name": "conservador", "candles": candles, "patterns": patterns,
     "config": BacktestConfig(risk_per_trade=0.01)},
    {"name": "agresivo", "candles": candles, "patterns": patterns,
     "config": BacktestConfig(risk_per_trade=0.03)},
])

# Comparar
comparison = runner.compare(results)  # ranking por return/sharpe/win_rate/PF
```

### Walk Forward

```python
from app.backtesting.validation import WalkForwardValidator
from app.backtesting.engine import BacktestEngine

def evaluate(test_candles):
    result = BacktestEngine().run(test_candles, patterns)
    return {
        "total_return": result.total_return,
        "sharpe_ratio": result.metrics.sharpe_ratio,
        "win_rate": result.metrics.win_rate,
    }

validator = WalkForwardValidator(
    train_size=300,   # velas de entrenamiento acumulado
    test_size=100,    # velas de test consecutivas
    step=100,         # desplazamiento entre folds
    evaluate_fn=evaluate,
)
result = validator.run(candles)
# result.aggregate -> {"sharpe_ratio": {"mean": ..., "std": ..., "min": ..., "max": ...}}
```

### Out Of Sample y Rolling Window

```python
from app.backtesting.validation import OutOfSampleValidator, RollingWindowValidator

# Fuera de muestra: últimos 30% como test
oos = OutOfSampleValidator(test_ratio=0.3, evaluate_fn=evaluate).run(candles)

# Rolling window: ventanas fijas deslizantes
rolling = RollingWindowValidator(window_size=200, step=50, evaluate_fn=evaluate).run(candles)
```

### Validación Cruzada

```python
from app.backtesting.validation import CrossValidator

cv = CrossValidator(n_splits=5, evaluate_fn=evaluate).run(candles)
# Agrega media/desviación de cada métrica entre folds
```

### Monte Carlo

```python
from app.backtesting.validation import MonteCarloSimulator

# Usar los trades de un backtest previo
simulator = MonteCarloSimulator(random_state=42)
mc = simulator.simulate(
    trades=result.trades,
    n_simulations=1000,
    initial_capital=100000,
)

mc.probability_of_profit   # % de simulaciones rentables
mc.percentiles             # {"p5": ..., "p50": ..., "p95": ...}
mc.var_95                  # Value at Risk (pérdida en el peor 5%)
mc.cvar_95                 # Conditional VaR (pérdida media del peor 5%)
mc.max_drawdowns           # drawdown por simulación
```

### Optimización de Parámetros

```python
from app.backtesting.optimization import BacktestOptimizer
from app.backtesting.engine import BacktestEngine
from app.backtesting.models import BacktestConfig

def objective(**params):
    cfg = BacktestConfig(**params)
    return BacktestEngine(cfg).run(candles, patterns).metrics.sharpe_ratio

optimizer = BacktestOptimizer(random_state=42)

# Grid search
grid = optimizer.grid_search({"risk_per_trade": [0.01, 0.02, 0.03]}, objective)

# Random search
random = optimizer.random_search({"risk_per_trade": [0.005, 0.01, 0.02, 0.03, 0.05]},
                                 objective, n_iter=50)

# Optimización bayesiana (Optuna / TPE)
bayes = optimizer.bayesian_optimization(
    {"risk_per_trade": [0.005, 0.01, 0.02, 0.03, 0.05]},
    objective, n_trials=100,
)

bayes["best_params"]  # mejores parámetros encontrados
bayes["best_score"]   # mejor métrica alcanzada
```

---

## Métricas de Resultado

### Métricas Automáticas (`MetricsCalculator`)

El motor calcula automáticamente sobre cada backtest:

| Métrica | Campo | Descripción |
|---------|-------|-------------|
| Win Rate | `win_rate` | % de operaciones ganadoras |
| Profit Factor | `profit_factor` | Beneficio bruto / pérdida bruta |
| Sharpe | `sharpe_ratio` | Retorno ajustado por riesgo (retornos por trade) |
| Sortino | `sortino_ratio` | Sharpe penalizando solo la volatilidad negativa |
| Calmar | `calmar_ratio` | Retorno anualizado / Max Drawdown |
| Ulcer Index | `ulcer_index` | Raíz de la media de drawdowns al cuadrado |
| Max Drawdown | `max_drawdown_pct` | Peor caída pico-a-valle (%) |
| Drawdown medio | `average_drawdown_pct` | Drawdown medio de la curva |
| Expectancy | `expectancy` | Valor esperado por operación ($) |
| Expectancy R | `expectancy_r` | Valor esperado en múltiplos de riesgo |
| Payoff Ratio | `payoff_ratio` | Ganancia media / pérdida media |
| Volatilidad | `annualized_volatility` | Volatilidad anualizada de la curva |

### Métricas de Clasificación

Para evaluar señales y modelos ML (WIN vs LOSS):

```python
from app.backtesting.metrics import MetricsCalculator

cm = MetricsCalculator.classification_metrics(
    y_true=[1, 0, 1, 1, 0],   # resultados reales
    y_pred=[1, 1, 0, 1, 0],   # predicciones
    y_proba=[0.9, 0.6, 0.3, 0.8, 0.4],
)

cm.precision            # precision
cm.recall               # recall
cm.f1_score             # F1
cm.roc_auc              # AUC-ROC
cm.pr_auc               # AUC-PR
cm.confusion_matrix     # [[TN, FP], [FN, TP]]
```

---

## Mejores Prácticas

1. **Datos suficientes**: Usar mínimo 1 año de datos
2. **Múltiples mercados**: Probar en diferentes condiciones
3. **Validación out-of-sample**: Separar datos de entrenamiento/test
4. **Costos realistas**: Incluir comisiones y slippage
5. **Gestión de riesgo**: Configurar stops adecuados
6. **Análisis estadístico**: No confiar en un solo backtest
7. **Documentar**: Guardar configuración y resultados
