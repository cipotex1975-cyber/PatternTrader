from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional
from uuid import UUID

from app.confirmation.engine import ConfirmationEngine
from app.core.events.bus import get_event_bus
from app.core.events.models import Event, EventType
from app.core.logger import get_logger
from app.data.providers.base import OHLCV, IDataProvider
from app.health.engine import HealthEngine
from app.lifecycle.engine import LifecycleEngine
from app.lifecycle.models import LifecycleState
from app.market.candles.models import Candle, CandleData
from app.market.indicators.calculator import IndicatorCalculator
from app.patterns.base_pattern import (
    BasePattern,
    PatternResult,
    PatternStatus,
    TradeDirection,
)
from app.patterns.registry import PatternRegistry
from app.scoring.engine import ScoringEngine
from app.signals.engine import SignalEngine
from app.signals.models import SignalPriority
from app.telegram.notifier import TelegramNotifier

logger = get_logger("PatternPipeline")

DataSource = Callable[[str, str], Awaitable[list[Candle]] | list[Candle]]


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

    Detección → Lifecycle → Health → Confirmación → Scoring → Señal → Telegram.
    Los detectores solo detectan estructura; este motor administra los estados.
    """

    def __init__(
        self,
        data_source: Optional[DataSource] = None,
        provider: IDataProvider | None = None,
        max_candles: int = 500,
    ) -> None:
        self._data_source = data_source
        self._provider = provider
        self._max_candles = max_candles

        self._indicator_calculator = IndicatorCalculator()
        self._lifecycle = LifecycleEngine()
        self._health = HealthEngine()
        self._confirmation = ConfirmationEngine()
        self._scoring = ScoringEngine()
        self._signal_engine = SignalEngine()
        self._telegram = TelegramNotifier()
        self._event_bus = get_event_bus()

        self._tracked: dict[UUID, TrackedPattern] = {}
        self._active_keys: set[tuple[str, str, str]] = set()
        self._detectors = PatternRegistry.get_all_instances()

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

    async def _detect_new(
        self,
        candles: list[Candle],
        symbol: str,
        timeframe: str,
        latest_indicators: dict[str, float],
    ) -> None:
        for detector in self._detectors:
            result = detector.detect(candles, symbol, timeframe)
            if result is None:
                continue

            key = (symbol, timeframe, result.pattern_name)
            if key in self._active_keys:
                continue

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

            report = await self._health.calculate(result, detector, candles, latest_indicators)
            result.update_health(report.health)
            result.metadata["health_report"] = report.model_dump()

            if not detector.validate(result, candles):
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

        signal = await self._signal_engine.create_signal(result, score_result, ml_probability)
        if signal is None:
            return

        result.transition(PatternStatus.SIGNAL_SENT)
        await self._lifecycle.update_pattern_status(
            result,
            PatternStatus.SIGNAL_SENT,
            reason=f"Score {score_result.total_score:.1f}",
        )

        if signal.priority == SignalPriority.CRITICAL:
            await self._telegram.send_signal(signal)
            await self._event_bus.publish(
                Event(
                    type=EventType.SIGNAL_SENT,
                    source="PatternPipeline",
                    data={
                        "signal_id": str(signal.id),
                        "symbol": signal.symbol,
                        "pattern_name": signal.pattern_name,
                        "score": signal.score,
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
                or levels.get("valley")
                or 0
            )
            stop = stop_candidate if stop_candidate and stop_candidate < entry else entry * 0.99
        else:
            entry = levels.get("neckline") or 0
            if entry == 0:
                return
            stop_candidate = max(
                levels.get("peak1", 0),
                levels.get("peak2", 0),
                levels.get("head", 0),
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
