from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import UUID

from app.core.config.settings import get_settings
from app.core.events.bus import get_event_bus
from app.core.events.models import Event, EventType
from app.core.logger import get_logger
from app.patterns.base_pattern import PatternResult
from app.scoring.models import ScoreResult
from app.signals.models import Signal, SignalPriority
from app.strategy.base import StrategySignal as StrategySignalModel

logger = get_logger("SignalEngine")


class SignalEngine:
    def __init__(self, repository: Optional[Any] = None) -> None:
        settings = get_settings()
        self._scoring_config = settings.patterns.scoring
        self._sent_signals: dict[str, datetime] = {}
        self._signals: dict[UUID, Signal] = {}
        self._repository = repository
        self._cooldown_minutes = self._scoring_config.cooldown_minutes
        self._dedup_store_path = settings.telegram.dedup_store_path
        self._event_bus = get_event_bus()
        self._load_dedup()

    async def create_signal(
        self,
        pattern: PatternResult,
        score: ScoreResult,
        ml_probability: Optional[float] = None,
        strategy_signal: Optional[StrategySignalModel] = None,
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

        entry_price = (strategy_signal.entry_price if strategy_signal else pattern.entry_price) or 0
        stop_loss = (strategy_signal.stop_loss if strategy_signal else pattern.stop_loss) or 0
        take_profit = (strategy_signal.take_profit if strategy_signal else pattern.take_profit) or 0

        if entry_price == 0 or stop_loss == 0 or take_profit == 0:
            logger.warning(f"Invalid price levels for signal: {pattern.pattern_name}")
            return None

        rr_ratio = (
            abs(take_profit - entry_price) / abs(entry_price - stop_loss)
            if abs(entry_price - stop_loss) > 0
            else 0
        )

        reasons = self._generate_reasons(pattern, score, ml_probability)
        if strategy_signal is not None:
            reasons.append(f"Strategy: {strategy_signal.strategy_name}")
            reasons.extend(strategy_signal.reasons)

        direction = (
            strategy_signal.direction
            if strategy_signal is not None
            else ("LONG" if take_profit > entry_price else "SHORT")
        )

        metadata: dict = {}
        if strategy_signal is not None:
            metadata["strategy"] = strategy_signal.strategy_name
            metadata["strategy_confidence"] = strategy_signal.confidence
            metadata["strategy_size"] = strategy_signal.size

        signal = Signal(
            symbol=pattern.symbol,
            timeframe=pattern.timeframe,
            pattern_name=pattern.pattern_name,
            direction=direction,
            priority=priority,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=rr_ratio,
            score=score.total_score,
            health=pattern.health,
            ml_probability=ml_probability,
            reasons=reasons,
            expires_at=datetime.utcnow() + timedelta(hours=self._scoring_config.signal_ttl_hours),
            metadata=metadata,
        )

        self._sent_signals[signal_key] = datetime.utcnow()
        self._signals[signal.id] = signal
        self._save_dedup()

        if self._repository is not None:
            await self._repository.add(signal)

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
        ml_probability: Optional[float],
    ) -> list[str]:
        reasons = []

        reasons.append(f"Pattern: {pattern.pattern_name} detected")
        reasons.append(f"Score: {score.total_score:.1f}/100 ({score.grade})")

        if ml_probability is not None and ml_probability >= 0.7:
            reasons.append(f"ML probability: {ml_probability:.0%}")

        for component in score.components:
            if component.score >= 70:
                reasons.append(f"{component.name}: {component.score:.0f}/100")

        if pattern.health >= 80:
            reasons.append(f"Pattern health: {pattern.health:.0f}%")

        return reasons

    def get_sent_signals(self) -> dict[str, datetime]:
        return self._sent_signals.copy()

    def get_signals(self) -> list[Signal]:
        return list(self._signals.values())

    def get_signal(self, signal_id: UUID) -> Optional[Signal]:
        return self._signals.get(signal_id)

    async def mark_sent(self, signal_id: UUID) -> Optional[Signal]:
        signal = self._signals.get(signal_id)
        if signal is None:
            return None
        if signal.is_expired:
            logger.warning(f"Signal {signal_id} expired, not sending")
            await self.mark_failed(signal_id, reason="signal expired")
            return None
        signal.mark_sent()
        if self._repository is not None:
            await self._repository.update_status(signal.id, signal.status, sent_at=signal.sent_at)
        return signal

    async def mark_delivered(self, signal_id: UUID) -> Optional[Signal]:
        signal = self._signals.get(signal_id)
        if signal is None:
            return None
        signal.mark_delivered()
        if self._repository is not None:
            await self._repository.update_status(signal.id, signal.status)
        return signal

    async def mark_failed(self, signal_id: UUID, reason: str = "") -> Optional[Signal]:
        signal = self._signals.get(signal_id)
        if signal is None:
            return None
        signal.mark_failed(reason)
        if self._repository is not None:
            await self._repository.update_status(signal.id, signal.status)
        return signal

    def clear_cooldown(self) -> None:
        self._sent_signals.clear()
        self._save_dedup()

    def _load_dedup(self) -> None:
        try:
            if not os.path.exists(self._dedup_store_path):
                return
            with open(self._dedup_store_path, "r") as f:
                raw = json.load(f)
            for key, value in raw.items():
                try:
                    self._sent_signals[key] = datetime.fromisoformat(str(value))
                except (ValueError, TypeError):
                    continue
            logger.debug(f"Loaded {len(self._sent_signals)} dedup entries")
        except (OSError, ValueError) as e:
            logger.warning(f"Could not load dedup store {self._dedup_store_path}: {e}")

    def _save_dedup(self) -> None:
        try:
            directory = os.path.dirname(self._dedup_store_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            payload = {key: value.isoformat() for key, value in self._sent_signals.items()}
            with open(self._dedup_store_path, "w") as f:
                json.dump(payload, f, indent=2)
        except OSError as e:
            logger.warning(f"Could not save dedup store {self._dedup_store_path}: {e}")
