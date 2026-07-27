from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np

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
                elif trade.direction == TradeDirection.SHORT and candle.data.high >= trade.stop_loss:
                    self._close_trade(trade, trade.stop_loss, candle.data.timestamp, "SL_HIT")
                    continue

            if trade.take_profit:
                if trade.direction == TradeDirection.LONG and candle.data.high >= trade.take_profit:
                    self._close_trade(trade, trade.take_profit, candle.data.timestamp, "TP_HIT")
                elif trade.direction == TradeDirection.SHORT and candle.data.low <= trade.take_profit:
                    self._close_trade(trade, trade.take_profit, candle.data.timestamp, "TP_HIT")

    def _check_new_entries(
        self, candle: Candle, pattern_map: dict, index: int
    ) -> None:
        open_count = sum(1 for t in self._trades if t.status == TradeStatus.OPEN)
        if open_count >= self._config.max_positions:
            return

        for pattern_id, pattern in pattern_map.items():
            if pattern.status.value not in ["CONFIRMED", "SIGNAL_SENT"]:
                continue

            if pattern.detected_at and candle.data.timestamp < pattern.detected_at:
                continue

            existing = any(
                t.metadata.get("pattern_id") == str(pattern_id) for t in self._trades
            )
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

        self._equity_curve.append({
            "timestamp": candle.data.timestamp.isoformat(),
            "equity": self._capital + open_pnl,
            "capital": self._capital,
            "open_pnl": open_pnl,
        })

    def _calculate_metrics(self) -> BacktestMetrics:
        closed_trades = [t for t in self._trades if t.status == TradeStatus.CLOSED]
        if not closed_trades:
            return BacktestMetrics()

        wins = [t for t in closed_trades if t.pnl > 0]
        losses = [t for t in closed_trades if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in closed_trades)
        win_rate = len(wins) / len(closed_trades) if closed_trades else 0

        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        pnls = [t.pnl for t in closed_trades]
        avg_pnl = np.mean(pnls) if pnls else 0
        std_pnl = np.std(pnls) if pnls else 1
        sharpe = (avg_pnl / std_pnl) * np.sqrt(252) if std_pnl > 0 else 0

        equity_values = [e["equity"] for e in self._equity_curve]
        if equity_values:
            peak = equity_values[0]
            max_dd = 0
            for eq in equity_values:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
        else:
            max_dd = 0

        return BacktestMetrics(
            total_trades=len(closed_trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd * self._config.initial_capital,
            max_drawdown_pct=max_dd * 100,
            average_win=avg_win,
            average_loss=avg_loss,
            expectancy=avg_pnl,
            total_pnl=total_pnl,
            total_pnl_pct=(total_pnl / self._config.initial_capital) * 100,
        )
