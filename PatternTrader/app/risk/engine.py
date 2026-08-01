from __future__ import annotations

from app.core.config.settings import get_settings
from app.core.logger import get_logger
from app.patterns.base_pattern import PatternResult
from app.risk.models import PositionSize, RiskAssessment, RiskLimits

logger = get_logger("RiskEngine")


class RiskEngine:
    def __init__(self, initial_capital: float = 100000.0) -> None:
        settings = get_settings()
        self._limits = RiskLimits(
            max_risk_per_trade=settings.risk.max_risk_per_trade,
            max_daily_risk=settings.risk.max_daily_risk,
            max_exposure_per_asset=settings.risk.max_exposure_per_asset,
            max_correlated_exposure=settings.risk.max_correlated_exposure,
        )
        self._capital = initial_capital
        self._daily_pnl = 0.0
        self._open_positions: dict[str, float] = {}

    def assess(
        self,
        pattern: PatternResult,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> RiskAssessment:
        position_size = self._calculate_position_size(
            entry_price, stop_loss, take_profit, pattern.symbol
        )

        warnings = self._check_warnings(position_size, pattern.symbol)
        recommendations = self._generate_recommendations(position_size, warnings)
        risk_score = self._calculate_risk_score(position_size, warnings)

        is_acceptable = (
            risk_score < 70
            and len(warnings) == 0
            and position_size.risk_pct <= self._limits.max_risk_per_trade
        )

        return RiskAssessment(
            symbol=pattern.symbol,
            timeframe=pattern.timeframe,
            pattern_name=pattern.pattern_name,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            is_acceptable=is_acceptable,
            risk_score=risk_score,
            warnings=warnings,
            recommendations=recommendations,
        )

    def _calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        symbol: str,
    ) -> PositionSize:
        risk_amount = self._capital * self._limits.max_risk_per_trade
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit == 0:
            size = 0.0
        else:
            size = risk_amount / risk_per_unit

        risk_pct = (risk_amount / self._capital) * 100 if self._capital > 0 else 0
        potential_reward = abs(take_profit - entry_price) * size
        risk_reward_ratio = (
            abs(take_profit - entry_price) / risk_per_unit if risk_per_unit > 0 else 0
        )

        return PositionSize(
            symbol=symbol,
            direction="LONG" if take_profit > entry_price else "SHORT",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            size=size,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            potential_reward=potential_reward,
            risk_reward_ratio=risk_reward_ratio,
            max_loss=risk_amount,
        )

    def _check_warnings(self, position: PositionSize, symbol: str) -> list[str]:
        warnings = []

        if position.risk_pct > self._limits.max_risk_per_trade * 100:
            warnings.append(f"Risk per trade ({position.risk_pct:.2f}%) exceeds limit")

        if position.risk_reward_ratio < 1.5:
            warnings.append(f"Low risk/reward ratio: {position.risk_reward_ratio:.2f}")

        current_exposure = self._open_positions.get(symbol, 0)
        new_exposure = current_exposure + position.size * position.entry_price
        exposure_pct = new_exposure / self._capital if self._capital > 0 else 0

        if exposure_pct > self._limits.max_exposure_per_asset:
            warnings.append(f"Asset exposure ({exposure_pct:.2%}) exceeds limit")

        if self._daily_pnl < -self._capital * self._limits.max_daily_risk:
            warnings.append("Daily risk limit reached")

        open_count = len(self._open_positions)
        if open_count >= self._limits.max_open_positions:
            warnings.append("Maximum open positions reached")

        return warnings

    def _generate_recommendations(self, position: PositionSize, warnings: list[str]) -> list[str]:
        recommendations = []

        if position.risk_reward_ratio < 2.0:
            recommendations.append("Consider wider take profit or tighter stop loss")

        if position.risk_pct > self._limits.max_risk_per_trade * 50:
            recommendations.append("Reduce position size to lower risk")

        if warnings:
            recommendations.append("Review risk warnings before proceeding")

        return recommendations

    def _calculate_risk_score(self, position: PositionSize, warnings: list[str]) -> float:
        score = 50.0

        if position.risk_reward_ratio >= 3.0:
            score -= 15
        elif position.risk_reward_ratio >= 2.0:
            score -= 10
        elif position.risk_reward_ratio >= 1.5:
            score -= 5
        else:
            score += 10

        score += len(warnings) * 10

        if position.risk_pct > self._limits.max_risk_per_trade * 100:
            score += 15

        return max(0.0, min(100.0, score))

    def update_daily_pnl(self, pnl: float) -> None:
        self._daily_pnl += pnl

    def register_position(self, symbol: str, size: float, price: float) -> None:
        self._open_positions[symbol] = self._open_positions.get(symbol, 0) + size * price

    def close_position(self, symbol: str, size: float, price: float, pnl: float) -> None:
        if symbol in self._open_positions:
            self._open_positions[symbol] -= size * price
            if self._open_positions[symbol] <= 0:
                del self._open_positions[symbol]
        self._daily_pnl += pnl

    def reset_daily(self) -> None:
        self._daily_pnl = 0.0

    def get_current_exposure(self) -> dict[str, float]:
        return {
            symbol: exposure / self._capital if self._capital > 0 else 0
            for symbol, exposure in self._open_positions.items()
        }
