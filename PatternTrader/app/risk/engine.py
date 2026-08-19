from __future__ import annotations

from app.core.config.settings import get_settings
from app.core.logger import get_logger
from app.patterns.base_pattern import PatternResult
from app.risk.models import PositionSize, RiskAssessment, RiskLimits

logger = get_logger("RiskEngine")

_DEFAULT_SECTOR = "default"


class RiskEngine:
    def __init__(
        self,
        initial_capital: float | None = None,
        symbol_sectors: dict[str, str] | None = None,
        correlations: dict[str, dict[str, float]] | None = None,
        correlation_threshold: float = 0.7,
    ) -> None:
        settings = get_settings()
        self._limits = RiskLimits(
            max_risk_per_trade=settings.risk.max_risk_per_trade,
            max_daily_risk=settings.risk.max_daily_risk,
            max_exposure_per_asset=settings.risk.max_exposure_per_asset,
            max_correlated_exposure=settings.risk.max_correlated_exposure,
        )
        if initial_capital is None:
            initial_capital = settings.backtesting.default_initial_capital
        self._capital = initial_capital
        self._daily_pnl = 0.0
        self._open_positions: dict[str, float] = {}
        self._symbol_sectors = symbol_sectors or {}
        self._correlations = correlations or {}
        self._correlation_threshold = correlation_threshold

    def set_capital(self, capital: float) -> None:
        """Sincroniza el capital usado para sizing/warnings (útil en backtests)."""
        self._capital = capital

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
            and position_size.risk_pct <= self._limits.max_risk_per_trade * 100
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

        # Cap de exposición nocional por activo: evita que una sola posición
        # concentre más de max_exposure_per_asset * capital (con sizing por
        # riesgo, el nocional es intrínsecamente grande).
        max_notional = self._capital * self._limits.max_exposure_per_asset
        if entry_price > 0 and size * entry_price > max_notional:
            size = max_notional / entry_price

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

        self._check_sector_exposure(position, warnings)
        self._check_correlated_exposure(position, warnings)

        if self._daily_pnl < -self._capital * self._limits.max_daily_risk:
            warnings.append("Daily risk limit reached")

        open_count = len(self._open_positions)
        if open_count >= self._limits.max_open_positions:
            warnings.append("Maximum open positions reached")

        return warnings

    def _check_sector_exposure(self, position: PositionSize, warnings: list[str]) -> None:
        if not self._symbol_sectors:
            return
        sector = self._symbol_sectors.get(position.symbol, _DEFAULT_SECTOR)
        sector_notional = position.size * position.entry_price
        for symbol, notional in self._open_positions.items():
            if self._symbol_sectors.get(symbol, _DEFAULT_SECTOR) == sector:
                sector_notional += notional
        sector_pct = sector_notional / self._capital if self._capital > 0 else 0
        if sector_pct > self._limits.max_correlated_exposure:
            warnings.append(f"Sector {sector} exposure ({sector_pct:.2%}) exceeds correlated limit")

    def _check_correlated_exposure(self, position: PositionSize, warnings: list[str]) -> None:
        related = self._get_correlated_symbols(position.symbol)
        if not related:
            return
        correlated_notional = position.size * position.entry_price
        for symbol in related:
            correlated_notional += self._open_positions.get(symbol, 0)
        correlated_pct = correlated_notional / self._capital if self._capital > 0 else 0
        if correlated_pct > self._limits.max_correlated_exposure:
            warnings.append(
                f"Correlated exposure ({correlated_pct:.2%}) exceeds "
                f"{self._limits.max_correlated_exposure:.0%} limit"
            )

    def _get_correlated_symbols(self, symbol: str) -> list[str]:
        symbol_corr = self._correlations.get(symbol, {})
        return sorted(
            other for other, corr in symbol_corr.items() if corr >= self._correlation_threshold
        )

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
