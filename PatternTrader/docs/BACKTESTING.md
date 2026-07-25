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

### Ejemplo Simple

```python
import asyncio
from app.backtesting.engine import BacktestEngine
from app.backtesting.models import BacktestConfig
from app.market.candles.models import Candle, CandleData
from app.patterns.base_pattern import PatternResult, PatternType, PatternStatus
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
from app.patterns.base_pattern import PatternResult, PatternType, PatternStatus
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

## Mejores Prácticas

1. **Datos suficientes**: Usar mínimo 1 año de datos
2. **Múltiples mercados**: Probar en diferentes condiciones
3. **Validación out-of-sample**: Separar datos de entrenamiento/test
4. **Costos realistas**: Incluir comisiones y slippage
5. **Gestión de riesgo**: Configurar stops adecuados
6. **Análisis estadístico**: No confiar en un solo backtest
7. **Documentar**: Guardar configuración y resultados
