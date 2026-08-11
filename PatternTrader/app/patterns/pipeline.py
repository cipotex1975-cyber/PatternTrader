from __future__ import annotations

import inspect
import re
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Awaitable, Callable, Optional
from uuid import UUID

from app.confirmation.engine import ConfirmationEngine
from app.core.config.settings import get_settings
from app.core.events.bus import get_event_bus
from app.core.events.models import Event, EventType
from app.core.logger import get_logger
from app.data.providers.base import OHLCV, IDataProvider
from app.health.engine import HealthEngine
from app.lifecycle.engine import LifecycleEngine
from app.lifecycle.models import LifecycleState
from app.market.candles.models import Candle, CandleData
from app.market.indicators.calculator import IndicatorCalculator
from app.ml.features import extract_technical_features, features_to_dict
from app.patterns.base_pattern import (
    BasePattern,
    PatternResult,
    PatternStatus,
    TradeDirection,
)
from app.patterns.hypothesis import PatternHypothesis
from app.patterns.registry import PatternRegistry
from app.risk.engine import RiskEngine
from app.scoring.engine import ScoringEngine
from app.signals.engine import SignalEngine
from app.signals.models import SignalPriority
from app.strategy.engine import StrategyEngine
from app.strategy.manager import StrategyManager
from app.telegram.notifier import TelegramNotifier

logger = get_logger("PatternPipeline")

DataSource = Callable[[str, str], Awaitable[list[Candle]] | list[Candle]]

_TIMEFRAME_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}

_PRIORITY_SCORES = {
    SignalPriority.LOW: 25,
    SignalPriority.MEDIUM: 50,
    SignalPriority.HIGH: 75,
    SignalPriority.CRITICAL: 100,
}


def timeframe_to_seconds(timeframe: str) -> int:
    """Convierte un timeframe (``1m``, ``4h``, ``1d``…) a segundos."""
    match = re.fullmatch(r"(\d+)\s*([smhdw])", str(timeframe).strip().lower())
    if not match:
        return 3600
    return int(match.group(1)) * _TIMEFRAME_SECONDS[match.group(2)]


def ohlcv_to_candle(ohlcv: OHLCV, symbol: str, timeframe: str) -> Candle:
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        data=CandleData(
            timestamp=ohlcv.timestamp,
            open=ohlcv.open,
            high=ohlcv.high,
            low=ohlcv.low,
            close=ohlcv.close,
            volume=ohlcv.volume,
        ),
    )


@dataclass
class TrackedPattern:
    detector: BasePattern
    result: PatternResult
    lifecycle_id: UUID


class PatternPipeline:
    """Orquesta el flujo completo de un patrón detectado.

    Detección → Lifecycle → Health → Confirmación → Scoring → Hipótesis →
    Estrategia → Señal → Telegram.
    Los detectores solo detectan estructura; este motor administra los estados
    y emite hipótesis que las estrategias evalúan antes de generar señal.
    """

    def __init__(
        self,
        data_source: Optional[DataSource] = None,
        provider: IDataProvider | None = None,
        max_candles: int = 500,
        strategies: Optional[list[str]] = None,
        strategy_params: Optional[dict] = None,
        learning_service: Optional[object] = None,
        lifecycle_repository: Optional[object] = None,
        signal_repository: Optional[object] = None,
        risk_engine: Optional[RiskEngine] = None,
        strategy_manager: Optional[StrategyManager] = None,
    ) -> None:
        self._data_source = data_source
        self._provider = provider
        self._max_candles = max_candles

        self._indicator_calculator = IndicatorCalculator()
        self._lifecycle = LifecycleEngine(repository=lifecycle_repository)
        self._health = HealthEngine()
        self._confirmation = ConfirmationEngine()
        self._risk = risk_engine or RiskEngine()
        self._scoring = ScoringEngine()
        if learning_service is not None:
            self._scoring.attach_knowledge(learning_service)
        self._signal_engine = SignalEngine(repository=signal_repository)
        self._strategy_manager = strategy_manager
        self._strategy_engine = (
            strategy_manager.engine
            if strategy_manager is not None
            else StrategyEngine(strategies=strategies, parameters=strategy_params)
        )
        self._telegram = TelegramNotifier()
        self._event_bus = get_event_bus()

        self._tracked: dict[UUID, TrackedPattern] = {}
        self._active_keys: set[tuple[str, str, str]] = set()
        self._detectors = PatternRegistry.get_all_instances()

        settings = get_settings()
        self._max_patterns_per_symbol = settings.patterns.lifecycle.max_patterns_per_symbol
        self._health_interval_seconds = settings.patterns.health.recalculate_interval_seconds
        self._last_health_calc: dict[UUID, float] = {}

    @property
    def lifecycle(self) -> LifecycleEngine:
        return self._lifecycle

    @property
    def tracked(self) -> dict[UUID, TrackedPattern]:
        return self._tracked

    def attach_provider(self, provider: IDataProvider) -> None:
        self._provider = provider

    async def process_symbol(
        self,
        symbol: str,
        timeframe: str,
        candles: Optional[list[Candle]] = None,
    ) -> dict:
        if candles is None:
            candles = await self._fetch_candles(symbol, timeframe)
        if not candles:
            return self.stats()

        candles = candles[-self._max_candles :]
        latest_indicators = self._indicator_calculator.get_latest_indicators(candles)

        await self._detect_new(candles, symbol, timeframe, latest_indicators)
        await self._update_tracked(candles, latest_indicators)

        await self._emit_candle_update(symbol, timeframe, candles)

        return self.stats()

    def stats(self) -> dict:
        return {
            "tracked": len(self._tracked),
            "active": sum(1 for t in self._tracked.values() if t.result.is_active),
            "expired": len(self._lifecycle.get_by_state(LifecycleState.EXPIRED)),
            "confirmed": len(self._lifecycle.get_by_state(LifecycleState.CONFIRMED)),
            "signals_sent": len(self._lifecycle.get_by_state(LifecycleState.SIGNAL_SENT)),
        }

    async def _fetch_candles(self, symbol: str, timeframe: str) -> list[Candle]:
        if self._data_source is not None:
            result = self._data_source(symbol, timeframe)
            if inspect.isawaitable(result):
                return await result
            return result

        if self._provider is None:
            return []

        try:
            raw = await self._provider.get_history(
                symbol=symbol,
                timeframe=timeframe,
                limit=self._max_candles,
            )
            return [ohlcv_to_candle(r, symbol, timeframe) for r in raw]
        except Exception as e:
            logger.error(f"Failed to fetch candles for {symbol} {timeframe}: {e}")
            return []

    async def _emit_candle_update(
        self, symbol: str, timeframe: str, candles: list[Candle]
    ) -> None:
        if not candles:
            return
        latest = candles[-1].data
        await self._event_bus.publish(
            Event(
                type=EventType.CANDLE_UPDATED,
                source="PatternPipeline",
                data={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": latest.timestamp.isoformat(),
                    "open": latest.open,
                    "high": latest.high,
                    "low": latest.low,
                    "close": latest.close,
                    "volume": latest.volume,
                },
            )
        )

    async def _detect_new(
        self,
        candles: list[Candle],
        symbol: str,
        timeframe: str,
        latest_indicators: dict[str, float],
    ) -> None:
        active_count = sum(
            1
            for t in self._tracked.values()
            if t.result.symbol == symbol and t.result.is_active
        )
        at_pattern_cap = (
            self._max_patterns_per_symbol > 0
            and active_count >= self._max_patterns_per_symbol
        )

        for detector in self._detectors:
            result = detector.detect(candles, symbol, timeframe)
            if result is None:
                continue

            key = (symbol, timeframe, result.pattern_name)
            if key in self._active_keys:
                continue

            if at_pattern_cap:
                logger.debug(
                    f"Skip {result.pattern_name} on {symbol}:{timeframe}: "
                    f"pattern cap {self._max_patterns_per_symbol} reached"
                )
                continue

            if result.expires_at is None:
                result.expires_at = result.detected_at + timedelta(
                    seconds=timeframe_to_seconds(timeframe) * result.max_confirmation_candles
                )

            lifecycle = await self._lifecycle.register(result)
            self._tracked[result.id] = TrackedPattern(
                detector=detector,
                result=result,
                lifecycle_id=lifecycle.id,
            )
            self._active_keys.add(key)

            result.metadata["detector"] = detector.name
            await self._event_bus.publish(
                Event(
                    type=EventType.PATTERN_DETECTED,
                    source="PatternPipeline",
                    data={
                        "pattern_id": str(result.id),
                        "pattern_name": result.pattern_name,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "confidence": result.confidence,
                    },
                )
            )

            logger.info(f"Detected {result.pattern_name} on {symbol}:{timeframe}")

    async def _update_tracked(
        self,
        candles: list[Candle],
        latest_indicators: dict[str, float],
    ) -> None:
        for pattern_id in list(self._tracked.keys()):
            tracked = self._tracked[pattern_id]
            result = tracked.result
            detector = tracked.detector

            detector.update(result, candles)

            self._prepare_price_levels(result)

            now = time.monotonic()
            last_calc = self._last_health_calc.get(pattern_id, 0.0)
            if (
                self._health_interval_seconds <= 0
                or now - last_calc >= self._health_interval_seconds
            ):
                report = await self._health.calculate(
                    result, detector, candles, latest_indicators
                )
                result.update_health(report.health)
                result.metadata["health_report"] = report.model_dump()
                self._last_health_calc[pattern_id] = now

            if not detector.validate(result, candles):
                lifecycle = self._lifecycle.get_by_pattern(result.id)
                if (
                    result.status == PatternStatus.SIGNAL_SENT
                    and lifecycle is not None
                    and lifecycle.current_state == LifecycleState.SIGNAL_SENT
                ):
                    result.transition(PatternStatus.CANCELLED)
                    result.metadata["cancellation_reason"] = (
                        "pattern deformation before trade entry"
                    )
                    await self._lifecycle.update_pattern_status(
                        result,
                        PatternStatus.CANCELLED,
                        "pattern deformation before trade entry",
                    )
                else:
                    detector.invalidate(result, reason="pattern deformation")
                    await self._lifecycle.update_pattern_status(
                        result, PatternStatus.INVALIDATED, "pattern deformation"
                    )

            if not result.is_active:
                await self._lifecycle.update_pattern_status(result, result.status)
                self._forget(pattern_id, result)
                continue

            confirm_result = None
            if result.status == PatternStatus.WAITING_BREAKOUT:
                confirm_result = self._confirmation.confirm(result, latest_indicators, candles)
                if confirm_result.is_confirmed:
                    result.transition(PatternStatus.CONFIRMED)
                    await self._event_bus.publish(
                        Event(
                            type=EventType.PATTERN_CONFIRMED,
                            source="PatternPipeline",
                            data={
                                "pattern_id": str(result.id),
                                "pattern_name": result.pattern_name,
                                "symbol": result.symbol,
                                "confirmation_score": confirm_result.score,
                            },
                        )
                    )
                    logger.info(
                        f"Confirmed {result.pattern_name} on "
                        f"{result.symbol}:{result.timeframe} (score {confirm_result.score:.1f})"
                    )
                else:
                    result.metadata["confirmation"] = confirm_result.model_dump()

            await self._advance_lifecycle(tracked)

            if result.status == PatternStatus.CONFIRMED:
                await self._score_and_signal(tracked, candles, latest_indicators)

    async def _advance_lifecycle(self, tracked: TrackedPattern) -> None:
        result = tracked.result
        status = result.status

        if status == PatternStatus.DETECTED:
            result.transition(PatternStatus.FORMING)
        elif status == PatternStatus.FORMING:
            if result.key_levels:
                result.transition(PatternStatus.WAITING_BREAKOUT)

        await self._lifecycle.update_pattern_status(result, result.status)

    async def _score_and_signal(
        self,
        tracked: TrackedPattern,
        candles: list[Candle],
        latest_indicators: dict[str, float],
    ) -> None:
        result = tracked.result
        score_result = self._scoring.calculate_score(result, latest_indicators, candles)
        result.score = score_result.total_score

        ml_probability = next(
            (c.score / 100.0 for c in score_result.components if c.name == "ml_history"),
            0.0,
        )

        confirmation = result.metadata.get("confirmation") or {}
        hypothesis = PatternHypothesis(
            pattern=result,
            indicators=latest_indicators,
            score=score_result,
            ml_probability=ml_probability,
            confirmation_score=confirmation.get("score"),
            candles=candles[-50:],
            market_structure=result.metadata.get("market_structure", {}) or {},
        )

        if self._strategy_manager is not None:
            strategy_result = self._strategy_manager.evaluate(hypothesis)
        else:
            strategy_result = self._strategy_engine.evaluate(hypothesis)
        result.metadata["strategy_decisions"] = strategy_result.to_dict()

        if not strategy_result.has_entry:
            logger.info(
                f"No strategy entry for {result.pattern_name} "
                f"{result.symbol}:{result.timeframe} (score {score_result.total_score:.1f})"
            )
            return

        best = strategy_result.best
        signal = await self._signal_engine.create_signal(
            result,
            score_result,
            ml_probability,
            strategy_signal=best.signal if best else None,
        )
        if signal is None:
            return

        risk_assessment = self._risk.assess(
            result, signal.entry_price, signal.stop_loss, signal.take_profit
        )
        result.metadata["risk_assessment"] = risk_assessment.model_dump(mode="json")
        if not risk_assessment.is_acceptable:
            reason = "; ".join(risk_assessment.warnings) or "risk limits exceeded"
            result.transition(PatternStatus.REJECTED)
            await self._lifecycle.update_pattern_status(
                result, PatternStatus.REJECTED, reason=f"Risk rejected: {reason}"
            )
            logger.info(
                f"Signal rejected by risk: {result.symbol} {result.pattern_name} "
                f"({reason})"
            )
            return

        result.transition(PatternStatus.SIGNAL_SENT)
        await self._lifecycle.update_pattern_status(
            result,
            PatternStatus.SIGNAL_SENT,
            reason=f"Score {score_result.total_score:.1f}",
        )

        min_priority = SignalPriority(get_settings().telegram.min_priority)
        if signal.priority_score >= _PRIORITY_SCORES[min_priority]:
            sent = await self._signal_engine.mark_sent(signal.id)
            if sent is None:
                return
            delivered = await self._telegram.send_signal(
                signal, candles=candles, pattern=result
            )
            if delivered:
                await self._signal_engine.mark_delivered(signal.id)
            else:
                await self._signal_engine.mark_failed(
                    signal.id, reason="telegram send failed"
                )
            features = extract_technical_features(candles)
            await self._event_bus.publish(
                Event(
                    type=EventType.SIGNAL_SENT,
                    source="PatternPipeline",
                    data={
                        "signal_id": str(signal.id),
                        "pattern_id": str(result.id),
                        "symbol": signal.symbol,
                        "timeframe": signal.timeframe,
                        "pattern_name": signal.pattern_name,
                        "direction": signal.direction,
                        "score": signal.score,
                        "entry_price": signal.entry_price,
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit,
                        "risk_reward_ratio": signal.risk_reward_ratio,
                        "strategy": signal.metadata.get("strategy", ""),
                        "size": signal.metadata.get("strategy_size"),
                        "indicators": features_to_dict(features),
                    },
                )
            )
            logger.info(
                f"Signal sent: {signal.symbol} {signal.pattern_name} " f"score {signal.score:.1f}"
            )
        else:
            logger.info(
                f"Signal prepared (not sent): {signal.symbol} {signal.pattern_name} "
                f"score {signal.score:.1f} priority {signal.priority.value}"
            )

    def _forget(self, pattern_id: UUID, result: PatternResult) -> None:
        self._tracked.pop(pattern_id, None)
        self._active_keys.discard((result.symbol, result.timeframe, result.pattern_name))
        self._last_health_calc.pop(pattern_id, None)

    def _prepare_price_levels(self, result: PatternResult) -> None:
        if (
            result.entry_price is not None
            and result.stop_loss is not None
            and result.take_profit is not None
            and result.risk_reward_ratio is not None
        ):
            return

        levels = result.key_levels
        if not levels:
            return

        default_rr = 2.0
        target = levels.get("target") or 0

        if result.direction == TradeDirection.LONG:
            entry = levels.get("neckline") or levels.get("pole_high") or levels.get("target") or 0
            if entry == 0:
                return
            stop_candidate = (
                levels.get("trough1")
                or levels.get("trough2")
                or levels.get("flag_low")
                or levels.get("pennant_low")
                or levels.get("valley")
                or 0
            )
            stop = stop_candidate if stop_candidate and stop_candidate < entry else entry * 0.99
        else:
            entry = levels.get("neckline") or levels.get("pole_low") or 0
            if entry == 0:
                return
            stop_candidate = max(
                levels.get("peak1", 0),
                levels.get("peak2", 0),
                levels.get("head", 0),
                levels.get("flag_high", 0),
                levels.get("pennant_high", 0),
            )
            stop = stop_candidate * 1.001 if stop_candidate > entry else entry * 1.01

        tp = self._resolve_take_profit(entry, stop, target, default_rr)

        result.entry_price = entry
        result.stop_loss = stop
        result.take_profit = tp
        risk = abs(entry - stop)
        if risk > 0:
            result.risk_reward_ratio = abs(tp - entry) / risk

    def _resolve_take_profit(
        self, entry: float, stop: float, target: float, default_rr: float
    ) -> float:
        risk = abs(entry - stop)
        if risk == 0:
            return entry * 0.99

        long = entry > stop
        if target:
            on_side = (long and target > entry) or (not long and target < entry)
            if on_side and abs(target - entry) / risk >= default_rr:
                return target

        if long:
            return entry + risk * default_rr
        return entry - risk * default_rr
