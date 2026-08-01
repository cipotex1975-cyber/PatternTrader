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

logger = get_logger("BacktestEngine")


class BacktestEngine:
    def __init__(self, config: BacktestConfig | None = None) -> None:
        self._config = config or BacktestConfig()
        self._trades: list[Trade] = []
        self._equity_curve: list[dict] = []
        self._capital = self._config.initial_capital

    def run(
        self,
        candles: list[Candle],
        patterns: list[PatternResult],
    ) -> BacktestResult:
        logger.info(f"Starting backtest with {len(candles)} candles and {len(patterns)} patterns")

        self._trades = []
        self._equity_curve = []
        self._capital = self._config.initial_capital

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

            if trade.stop_loss:
                if trade.direction == TradeDirection.LONG and candle.data.low <= trade.stop_loss:
                    self._close_trade(trade, trade.stop_loss, candle.data.timestamp, "SL_HIT")
                    continue
                elif (
                    trade.direction == TradeDirection.SHORT and candle.data.high >= trade.stop_loss
                ):
                    self._close_trade(trade, trade.stop_loss, candle.data.timestamp, "SL_HIT")
                    continue

            if trade.take_profit:
                if trade.direction == TradeDirection.LONG and candle.data.high >= trade.take_profit:
                    self._close_trade(trade, trade.take_profit, candle.data.timestamp, "TP_HIT")
                elif (
                    trade.direction == TradeDirection.SHORT and candle.data.low <= trade.take_profit
                ):
                    self._close_trade(trade, trade.take_profit, candle.data.timestamp, "TP_HIT")

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

            self._open_trade(pattern, candle)

    def _open_trade(self, pattern: PatternResult, candle: Candle) -> None:
        entry_price = pattern.entry_price if pattern.entry_price else candle.data.close
        stop_loss = pattern.stop_loss
        take_profit = pattern.take_profit

        if stop_loss is None or take_profit is None:
            return

        trade_size = self._calculate_position_size(entry_price, stop_loss)

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
        logger.debug(f"Opened trade: {trade.symbol} at {trade.entry_price}")

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
        trade.pnl -= commission

        self._capital += trade.pnl
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

    def _calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        risk_amount = self._capital * self._config.risk_per_trade
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit == 0:
            return 0.0

        return risk_amount / risk_per_unit

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
