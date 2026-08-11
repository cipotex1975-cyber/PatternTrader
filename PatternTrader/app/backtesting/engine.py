from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.backtesting.metrics import MetricsCalculator
from app.backtesting.models import (
    BacktestConfig,
    BacktestMetrics,
    BacktestResult,
    Trade,
    TradeDirection,
    TradeStatus,
)
from app.core.logger import get_logger
from app.market.candles.models import Candle
from app.patterns.base_pattern import PatternResult
from app.risk.engine import RiskEngine

logger = get_logger("BacktestEngine")


class BacktestEngine:
    def __init__(
        self,
        config: BacktestConfig | None = None,
        risk_engine: RiskEngine | None = None,
    ) -> None:
        self._config = config or BacktestConfig()
        self._trades: list[Trade] = []
        self._equity_curve: list[dict] = []
        self._capital = self._config.initial_capital
        self._risk = risk_engine or RiskEngine(
            initial_capital=self._config.initial_capital
        )
        self._candles: list[Candle] = []

    def run(
        self,
        candles: list[Candle],
        patterns: list[PatternResult],
    ) -> BacktestResult:
        logger.info(f"Starting backtest with {len(candles)} candles and {len(patterns)} patterns")

        self._trades = []
        self._equity_curve = []
        self._capital = self._config.initial_capital
        self._candles = candles
        self._risk = RiskEngine(initial_capital=self._capital)
        self._risk.set_capital(self._capital)

        pattern_map = {p.id: p for p in patterns}

        for i, candle in enumerate(candles):
            self._update_open_trades(candle, i)
            self._check_new_entries(candle, pattern_map, i)
            self._record_equity(candle)

        self._close_all_open_trades(candles[-1] if candles else None)

        metrics = self._calculate_metrics()

        return BacktestResult(
            config=self._config,
            metrics=metrics,
            trades=self._trades,
            equity_curve=self._equity_curve,
            start_date=candles[0].data.timestamp if candles else datetime.now(timezone.utc),
            end_date=candles[-1].data.timestamp if candles else datetime.now(timezone.utc),
            initial_capital=self._config.initial_capital,
            final_capital=self._capital,
        )

    def _update_open_trades(self, candle: Candle, index: int) -> None:
        for trade in self._trades:
            if trade.status != TradeStatus.OPEN:
                continue

            # Trailing stop update
            if self._config.use_trailing_stop:
                self._apply_trailing_stop(trade, candle)

            if trade.stop_loss:
                if trade.direction == TradeDirection.LONG and candle.data.low <= trade.stop_loss:
                    reason = "TRAILING_STOP" if trade.metadata.get("trailing_active") else "SL_HIT"
                    self._close_trade(trade, trade.stop_loss, candle.data.timestamp, reason)
                    continue
                elif (
                    trade.direction == TradeDirection.SHORT and candle.data.high >= trade.stop_loss
                ):
                    reason = "TRAILING_STOP" if trade.metadata.get("trailing_active") else "SL_HIT"
                    self._close_trade(trade, trade.stop_loss, candle.data.timestamp, reason)
                    continue

            if trade.take_profit:
                if trade.direction == TradeDirection.LONG and candle.data.high >= trade.take_profit:
                    self._close_trade(trade, trade.take_profit, candle.data.timestamp, "TP_HIT")
                elif (
                    trade.direction == TradeDirection.SHORT and candle.data.low <= trade.take_profit
                ):
                    self._close_trade(trade, trade.take_profit, candle.data.timestamp, "TP_HIT")

    def _apply_trailing_stop(self, trade: Trade, candle: Candle) -> None:
        pct = self._config.trailing_stop_pct
        if trade.direction == TradeDirection.LONG:
            high_water = max(
                trade.metadata.get("high_watermark", trade.entry_price),
                candle.data.high,
            )
            trade.metadata["high_watermark"] = high_water
            trailing_stop = high_water * (1.0 - pct)
            if trade.stop_loss is None or trailing_stop > trade.stop_loss:
                trade.stop_loss = trailing_stop
                trade.metadata["trailing_active"] = True
        else:
            low_water = min(
                trade.metadata.get("low_watermark", trade.entry_price),
                candle.data.low,
            )
            trade.metadata["low_watermark"] = low_water
            trailing_stop = low_water * (1.0 + pct)
            if trade.stop_loss is None or trailing_stop < trade.stop_loss:
                trade.stop_loss = trailing_stop
                trade.metadata["trailing_active"] = True

    def _check_new_entries(self, candle: Candle, pattern_map: dict, index: int) -> None:
        open_count = sum(1 for t in self._trades if t.status == TradeStatus.OPEN)
        if open_count >= self._config.max_positions:
            return

        for pattern_id, pattern in pattern_map.items():
            if pattern.status.value not in ["CONFIRMED", "SIGNAL_SENT"]:
                continue

            if pattern.detected_at and candle.data.timestamp < pattern.detected_at:
                continue

            existing = any(t.metadata.get("pattern_id") == str(pattern_id) for t in self._trades)
            if existing:
                continue

            self._open_trade(pattern, candle, index)

    def _open_trade(self, pattern: PatternResult, candle: Candle, index: int) -> None:
        entry_price = pattern.entry_price if pattern.entry_price else candle.data.close
        stop_loss = pattern.stop_loss
        take_profit = pattern.take_profit

        if self._config.use_atr_stops:
            atr = self._calculate_atr(index, self._config.atr_period)
            if atr > 0:
                direction = pattern.direction.value if pattern.direction else "LONG"
                if direction == "LONG":
                    stop_loss = entry_price - atr * self._config.atr_sl_multiplier
                    take_profit = entry_price + atr * self._config.atr_tp_multiplier
                else:
                    stop_loss = entry_price + atr * self._config.atr_sl_multiplier
                    take_profit = entry_price - atr * self._config.atr_tp_multiplier

        if stop_loss is None or take_profit is None:
            return

        self._risk.set_capital(self._capital)
        assessment = self._risk.assess(
            pattern,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        if not assessment.is_acceptable:
            logger.debug(
                f"Trade rejected by RiskEngine: {pattern.symbol} "
                f"warnings={assessment.warnings}"
            )
            return

        trade_size = assessment.position_size.size
        if trade_size <= 0:
            return

        trade = Trade(
            id=str(uuid.uuid4()),
            symbol=pattern.symbol,
            timeframe=pattern.timeframe,
            direction=TradeDirection(pattern.direction.value),
            entry_price=entry_price,
            entry_time=candle.data.timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            size=trade_size,
            pattern_name=pattern.pattern_name,
            score=pattern.score,
            metadata={"pattern_id": str(pattern.id)},
        )

        self._trades.append(trade)
        self._risk.register_position(trade.symbol, trade.size, trade.entry_price)
        logger.debug(f"Opened trade: {trade.symbol} at {trade.entry_price}")

    def _calculate_atr(self, index: int, period: int) -> float:
        if not self._candles or index < 1:
            return 0.0
        start = max(0, index - period)
        tr_list: list[float] = []
        for i in range(start + 1, index + 1):
            curr = self._candles[i].data
            prev_close = self._candles[i - 1].data.close
            tr = max(
                curr.high - curr.low,
                abs(curr.high - prev_close),
                abs(curr.low - prev_close),
            )
            tr_list.append(tr)
        return float(sum(tr_list) / len(tr_list)) if tr_list else 0.0

    def _close_trade(
        self, trade: Trade, exit_price: float, exit_time: datetime, reason: str
    ) -> None:
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.status = TradeStatus.CLOSED

        if trade.direction == TradeDirection.LONG:
            trade.pnl = (exit_price - trade.entry_price) * trade.size
            trade.pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.size
            trade.pnl_pct = (trade.entry_price - exit_price) / trade.entry_price

        commission = abs(trade.pnl) * self._config.commission
        trade.fees = commission
        trade.pnl -= commission

        self._capital += trade.pnl
        self._risk.close_position(
            trade.symbol, trade.size, trade.entry_price, trade.pnl
        )
        self._risk.set_capital(self._capital)
        trade.metadata["close_reason"] = reason

        logger.debug(f"Closed trade: {trade.symbol} PnL: {trade.pnl:.2f}")

    def _close_all_open_trades(self, last_candle: Candle | None) -> None:
        if not last_candle:
            return

        for trade in self._trades:
            if trade.status == TradeStatus.OPEN:
                self._close_trade(
                    trade, last_candle.data.close, last_candle.data.timestamp, "BACKTEST_END"
                )

    def _record_equity(self, candle: Candle) -> None:
        open_pnl = 0.0
        for trade in self._trades:
            if trade.status == TradeStatus.OPEN:
                if trade.direction == TradeDirection.LONG:
                    open_pnl += (candle.data.close - trade.entry_price) * trade.size
                else:
                    open_pnl += (trade.entry_price - candle.data.close) * trade.size

        self._equity_curve.append(
            {
                "timestamp": candle.data.timestamp.isoformat(),
                "equity": self._capital + open_pnl,
                "capital": self._capital,
                "open_pnl": open_pnl,
            }
        )

    def _calculate_metrics(self) -> BacktestMetrics:
        closed_trades = [t for t in self._trades if t.status == TradeStatus.CLOSED]
        if not closed_trades:
            return BacktestMetrics()

        start = datetime.fromisoformat(self._equity_curve[0]["timestamp"])
        end = datetime.fromisoformat(self._equity_curve[-1]["timestamp"])

        return MetricsCalculator.calculate(
            trades=closed_trades,
            equity_curve=self._equity_curve,
            initial_capital=self._config.initial_capital,
            start_date=start,
            end_date=end,
        )
