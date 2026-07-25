from __future__ import annotations

from typing import Optional

from app.core.config.settings import get_settings
from app.core.logger import get_logger
from app.market.candles.models import Candle
from app.patterns.base_pattern import PatternResult
from app.confirmation.models import (
    ConfirmationCheck,
    ConfirmationResult,
    ConfirmationRule,
    ConfirmationStatus,
)

logger = get_logger("ConfirmationEngine")


class ConfirmationEngine:
    def __init__(self) -> None:
        self._rules = self._initialize_rules()

    def _initialize_rules(self) -> list[ConfirmationRule]:
        return [
            ConfirmationRule(
                name="breakout",
                description="Price has broken through key level",
                required=True,
                weight=1.0,
            ),
            ConfirmationRule(
                name="volume_confirmation",
                description="Volume confirms the breakout",
                required=True,
                weight=0.8,
            ),
            ConfirmationRule(
                name="atr_sufficient",
                description="ATR is sufficient for the move",
                required=False,
                weight=0.6,
            ),
            ConfirmationRule(
                name="trend_alignment",
                description="Pattern aligns with higher timeframe trend",
                required=False,
                weight=0.7,
            ),
            ConfirmationRule(
                name="risk_reward",
                description="Risk/Reward ratio is acceptable",
                required=True,
                weight=0.9,
            ),
            ConfirmationRule(
                name="spread_acceptable",
                description="Spread is within acceptable range",
                required=False,
                weight=0.4,
            ),
        ]

    def confirm(
        self,
        pattern: PatternResult,
        indicators: dict[str, float],
        candles: list[Candle],
    ) -> ConfirmationResult:
        checks: list[ConfirmationCheck] = []

        breakout_check = self._check_breakout(pattern, candles)
        checks.append(breakout_check)

        volume_check = self._check_volume(indicators, candles)
        checks.append(volume_check)

        atr_check = self._check_atr(indicators, candles)
        checks.append(atr_check)

        trend_check = self._check_trend_alignment(indicators, pattern)
        checks.append(trend_check)

        rr_check = self._check_risk_reward(pattern)
        checks.append(rr_check)

        spread_check = self._check_spread(indicators)
        checks.append(spread_check)

        passed = sum(1 for c in checks if c.status == ConfirmationStatus.PASSED)
        failed = sum(1 for c in checks if c.status == ConfirmationStatus.FAILED)

        required_rules = [r for r in self._rules if r.required]
        required_checks = [c for c in checks if c.rule.required]
        passed_required = sum(1 for c in required_checks if c.status == ConfirmationStatus.PASSED)

        score = self._calculate_score(checks)
        is_confirmed = passed_required == len(required_rules) and score >= 60

        return ConfirmationResult(
            is_confirmed=is_confirmed,
            score=score,
            checks=checks,
            passed_checks=passed,
            failed_checks=failed,
            total_required=len(required_rules),
            passed_required=passed_required,
            metadata={
                "pattern_name": pattern.pattern_name,
                "symbol": pattern.symbol,
                "timeframe": pattern.timeframe,
            },
        )

    def _check_breakout(
        self, pattern: PatternResult, candles: list[Candle]
    ) -> ConfirmationCheck:
        rule = self._get_rule("breakout")
        if not candles:
            return ConfirmationCheck(
                rule=rule,
                status=ConfirmationStatus.FAILED,
                message="No candles data available",
            )

        latest_close = candles[-1].data.close
        neckline = pattern.key_levels.get("neckline", 0)

        if neckline == 0:
            return ConfirmationCheck(
                rule=rule,
                status=ConfirmationStatus.FAILED,
                message="No neckline defined",
            )

        if pattern.pattern_type.value == "reversal":
            if "double_top" in pattern.pattern_name or "head_and_shoulders" in pattern.pattern_name:
                passed = latest_close < neckline
            else:
                passed = latest_close > neckline
        else:
            passed = True

        return ConfirmationCheck(
            rule=rule,
            status=ConfirmationStatus.PASSED if passed else ConfirmationStatus.FAILED,
            value=latest_close,
            threshold=neckline,
            message=f"Close {latest_close} vs Neckline {neckline}",
        )

    def _check_volume(
        self, indicators: dict[str, float], candles: list[Candle]
    ) -> ConfirmationCheck:
        rule = self._get_rule("volume_confirmation")
        if not candles or len(candles) < 20:
            return ConfirmationCheck(
                rule=rule,
                status=ConfirmationStatus.PENDING,
                message="Insufficient volume data",
            )

        recent_volume = sum(c.data.volume for c in candles[-5:]) / 5
        avg_volume = sum(c.data.volume for c in candles[-20:]) / 20

        if avg_volume == 0:
            return ConfirmationCheck(
                rule=rule,
                status=ConfirmationStatus.FAILED,
                message="Average volume is zero",
            )

        ratio = recent_volume / avg_volume
        passed = ratio > 1.2

        return ConfirmationCheck(
            rule=rule,
            status=ConfirmationStatus.PASSED if passed else ConfirmationStatus.FAILED,
            value=ratio,
            threshold=1.2,
            message=f"Volume ratio: {ratio:.2f}",
        )

    def _check_atr(
        self, indicators: dict[str, float], candles: list[Candle]
    ) -> ConfirmationCheck:
        rule = self._get_rule("atr_sufficient")
        atr = indicators.get("atr", 0)

        if atr == 0:
            return ConfirmationCheck(
                rule=rule,
                status=ConfirmationStatus.PENDING,
                message="ATR not available",
            )

        if candles:
            avg_price = sum(c.data.close for c in candles[-20:]) / 20
            if avg_price > 0:
                atr_pct = (atr / avg_price) * 100
                passed = 0.5 < atr_pct < 3.0
                return ConfirmationCheck(
                    rule=rule,
                    status=ConfirmationStatus.PASSED if passed else ConfirmationStatus.FAILED,
                    value=atr_pct,
                    threshold=1.5,
                    message=f"ATR%: {atr_pct:.2f}",
                )

        return ConfirmationCheck(
            rule=rule,
            status=ConfirmationStatus.PASSED,
            message="ATR check passed",
        )

    def _check_trend_alignment(
        self, indicators: dict[str, float], pattern: PatternResult
    ) -> ConfirmationCheck:
        rule = self._get_rule("trend_alignment")
        ema_21 = indicators.get("ema_21", 0)
        ema_50 = indicators.get("ema_50", 0)
        ema_200 = indicators.get("ema_200", 0)

        if ema_21 == 0 or ema_50 == 0:
            return ConfirmationCheck(
                rule=rule,
                status=ConfirmationStatus.PENDING,
                message="EMA data not available",
            )

        if pattern.pattern_type.value == "continuation":
            passed = ema_21 > ema_50
        elif pattern.pattern_type.value == "reversal":
            passed = True
        else:
            passed = True

        return ConfirmationCheck(
            rule=rule,
            status=ConfirmationStatus.PASSED if passed else ConfirmationStatus.FAILED,
            value=ema_21,
            threshold=ema_50,
            message=f"EMA21: {ema_21}, EMA50: {ema_50}",
        )

    def _check_risk_reward(self, pattern: PatternResult) -> ConfirmationCheck:
        rule = self._get_rule("risk_reward")
        rr = pattern.risk_reward_ratio

        if rr is None:
            return ConfirmationCheck(
                rule=rule,
                status=ConfirmationStatus.PENDING,
                message="R/R ratio not calculated",
            )

        passed = rr >= 2.0

        return ConfirmationCheck(
            rule=rule,
            status=ConfirmationStatus.PASSED if passed else ConfirmationStatus.FAILED,
            value=rr,
            threshold=2.0,
            message=f"R/R ratio: {rr:.2f}",
        )

    def _check_spread(self, indicators: dict[str, float]) -> ConfirmationCheck:
        rule = self._get_rule("spread_acceptable")
        return ConfirmationCheck(
            rule=rule,
            status=ConfirmationStatus.PASSED,
            message="Spread check passed (placeholder)",
        )

    def _get_rule(self, name: str) -> ConfirmationRule:
        for rule in self._rules:
            if rule.name == name:
                return rule
        return ConfirmationRule(name=name, description="Unknown rule")

    def _calculate_score(self, checks: list[ConfirmationCheck]) -> float:
        if not checks:
            return 0.0

        total_weight = sum(c.rule.weight for c in checks)
        if total_weight == 0:
            return 0.0

        score = 0.0
        for check in checks:
            if check.status == ConfirmationStatus.PASSED:
                score += check.rule.weight * 100
            elif check.status == ConfirmationStatus.PENDING:
                score += check.rule.weight * 50

        return score / total_weight
