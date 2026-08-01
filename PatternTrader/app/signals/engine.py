from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from app.core.config.settings import get_settings
from app.core.events.bus import get_event_bus
from app.core.events.models import Event, EventType
from app.core.logger import get_logger
from app.patterns.base_pattern import PatternResult
from app.scoring.models import ScoreResult
from app.signals.models import Signal, SignalPriority

logger = get_logger("SignalEngine")


class SignalEngine:
    def __init__(self) -> None:
        settings = get_settings()
        self._scoring_config = settings.patterns.scoring
        self._sent_signals: dict[str, datetime] = {}
        self._cooldown_minutes = 5
        self._event_bus = get_event_bus()

    async def create_signal(
        self,
        pattern: PatternResult,
        score: ScoreResult,
        ml_probability: float = 0.0,
    ) -> Optional[Signal]:
        signal_key = f"{pattern.symbol}:{pattern.pattern_name}:{pattern.timeframe}"

        if signal_key in self._sent_signals:
            last_sent = self._sent_signals[signal_key]
            if datetime.utcnow() - last_sent < timedelta(minutes=self._cooldown_minutes):
                logger.debug(f"Signal {signal_key} is in cooldown")
                return None

        priority = self._determine_priority(score.total_score)
        if priority is None:
            logger.debug(f"Score {score.total_score} too low for signal")
            return None

        entry_price = pattern.entry_price or 0
        stop_loss = pattern.stop_loss or 0
        take_profit = pattern.take_profit or 0

        if entry_price == 0 or stop_loss == 0 or take_profit == 0:
            logger.warning(f"Invalid price levels for signal: {pattern.pattern_name}")
            return None

        rr_ratio = (
            abs(take_profit - entry_price) / abs(entry_price - stop_loss)
            if abs(entry_price - stop_loss) > 0
            else 0
        )

        reasons = self._generate_reasons(pattern, score, ml_probability)

        signal = Signal(
            symbol=pattern.symbol,
            timeframe=pattern.timeframe,
            pattern_name=pattern.pattern_name,
            direction="LONG" if take_profit > entry_price else "SHORT",
            priority=priority,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=rr_ratio,
            score=score.total_score,
            health=pattern.health,
            ml_probability=ml_probability,
            reasons=reasons,
            expires_at=datetime.utcnow() + timedelta(hours=24),
        )

        self._sent_signals[signal_key] = datetime.utcnow()

        await self._event_bus.publish(
            Event(
                type=EventType.SIGNAL_CREATED,
                source="SignalEngine",
                data={
                    "signal_id": str(signal.id),
                    "symbol": signal.symbol,
                    "pattern": signal.pattern_name,
                    "score": signal.score,
                    "priority": signal.priority.value,
                },
            )
        )

        logger.info(
            f"Signal created: {signal.symbol} {signal.pattern_name} "
            f"Score: {signal.score} Priority: {signal.priority.value}"
        )

        return signal

    def _determine_priority(self, score: float) -> Optional[SignalPriority]:
        if score >= self._scoring_config.min_score_to_send:
            return SignalPriority.CRITICAL
        elif score >= self._scoring_config.min_score_to_alert:
            return SignalPriority.HIGH
        elif score >= self._scoring_config.min_score_to_prepare:
            return SignalPriority.MEDIUM
        elif score >= self._scoring_config.min_score_to_observe:
            return SignalPriority.LOW
        return None

    def _generate_reasons(
        self,
        pattern: PatternResult,
        score: ScoreResult,
        ml_probability: float,
    ) -> list[str]:
        reasons = []

        reasons.append(f"Pattern: {pattern.pattern_name} detected")
        reasons.append(f"Score: {score.total_score:.1f}/100 ({score.grade})")

        if ml_probability >= 0.7:
            reasons.append(f"ML probability: {ml_probability:.0%}")

        for component in score.components:
            if component.score >= 70:
                reasons.append(f"{component.name}: {component.score:.0f}/100")

        if pattern.health >= 80:
            reasons.append(f"Pattern health: {pattern.health:.0f}%")

        return reasons

    def get_sent_signals(self) -> dict[str, datetime]:
        return self._sent_signals.copy()

    def clear_cooldown(self) -> None:
        self._sent_signals.clear()
