from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional

from app.backtesting.models import Trade
from app.core.events.bus import get_event_bus
from app.core.events.models import Event, EventType
from app.core.logger import get_logger
from app.learning.features import FeatureBuilder
from app.learning.models import KnowledgeEntry, LearningMode, TradeOutcome
from app.learning.offline import OfflineLearner
from app.learning.online import OnlineLearner
from app.learning.repository import KnowledgeRepository
from app.ml.base import MLPrediction

logger = get_logger("LearningService")


class LearningService:
    """Aprende de cada operación cerrada y alimenta la base de conocimiento.

    - Modo OFFLINE: reentrena el modelo sobre toda la base al cerrar operaciones.
    - Modo ONLINE: actualiza el modelo de forma incremental por operación.
    """

    def __init__(
        self,
        repository: Optional[KnowledgeRepository] = None,
        mode: LearningMode = LearningMode.OFFLINE,
        feature_builder: Optional[FeatureBuilder] = None,
        offline_learner: Optional[OfflineLearner] = None,
        online_learner: Optional[OnlineLearner] = None,
        models_dir: str = "models",
        min_samples: int = 10,
        retrain_every: int = 10,
    ) -> None:
        self._repo = repository or KnowledgeRepository()
        self._mode = mode
        self._feature_builder = feature_builder or FeatureBuilder()
        model_path = os.path.join(models_dir, "knowledge_model.joblib")
        self._offline = offline_learner or OfflineLearner(
            feature_builder=self._feature_builder, model_path=model_path
        )
        self._online = online_learner or OnlineLearner(feature_builder=self._feature_builder)
        self._bus = get_event_bus()
        self._min_samples = min_samples
        self._retrain_every = retrain_every
        self._started = False

    @property
    def mode(self) -> LearningMode:
        return self._mode

    def set_mode(self, mode: LearningMode) -> None:
        self._mode = mode
        logger.info(f"LearningService modo cambiado a {mode.value}")

    @property
    def repository(self) -> Any:
        return self._repo

    async def start(self) -> None:
        if self._started:
            return
        self._bus.subscribe(EventType.TRADE_CLOSED, self._on_trade_closed)
        self._bus.subscribe(EventType.TRADE_OPENED, self._on_trade_opened)
        self._started = True
        logger.info(f"LearningService iniciado en modo {self._mode.value}")

    async def stop(self) -> None:
        if not self._started:
            return
        self._bus.unsubscribe(EventType.TRADE_CLOSED, self._on_trade_closed)
        self._bus.unsubscribe(EventType.TRADE_OPENED, self._on_trade_opened)
        self._started = False
        logger.info("LearningService detenido")

    async def _on_trade_closed(self, event: Event) -> None:
        trade = self._trade_from_event(event)
        if trade is None:
            return
        await self.record_trade(trade)

    async def _on_trade_opened(self, event: Event) -> None:
        trade = self._trade_from_event(event)
        if trade is None:
            return
        entry = self._entry_from_trade(trade)
        await self._repo.add(entry)
        logger.debug(f"Operación registrada en la base de conocimiento: {trade.symbol}")

    async def record_trade(
        self,
        trade: Trade | dict[str, Any],
        indicators: Optional[dict[str, float]] = None,
        variables: Optional[dict[str, Any]] = None,
        image_path: str = "",
    ) -> KnowledgeEntry:
        """Registra una operación cerrada y alimenta el modelo según el modo."""
        entry = self._entry_from_trade(trade)
        if indicators:
            entry.indicators.update(indicators)
        if variables:
            entry.variables.update(variables)
        if image_path:
            entry.image_path = image_path
        entry.ml_features = self._feature_builder.build(entry.indicators, entry.variables)

        await self._repo.add(entry)

        if self._mode == LearningMode.ONLINE:
            self._online.update(entry.ml_features, entry.is_win)
            logger.debug(
                f"Online update: {entry.pattern} -> {entry.outcome.value} "
                f"(muestras: {self._online.samples_seen})"
            )
        else:
            await self._retrain_offline_if_enough()

        logger.info(
            f"Operación registrada: {entry.instrument} {entry.pattern} "
            f"{entry.outcome.value} pnl={entry.pnl:.2f}"
        )
        return entry

    async def _retrain_offline_if_enough(self) -> None:
        count = await self._repo.count()
        if count >= self._min_samples and count % self._retrain_every == 0:
            await self.train_offline()

    async def train_offline(self, n_splits: int = 5) -> dict[str, Any]:
        entries = await self._repo.get_all()
        report = self._offline.train(entries, n_splits=n_splits)
        await self._bus.publish(
            Event(
                type=EventType.ML_TRAINED,
                source="learning_service",
                data={"model": "knowledge_model", "samples": len(entries), **report},
            )
        )
        return report

    def predict(
        self,
        indicators: dict[str, float],
        variables: Optional[dict[str, Any]] = None,
        instrument: str = "",
        timeframe: str = "",
        pattern: str = "",
    ) -> MLPrediction:
        """Predice la probabilidad de éxito usando el mejor modelo disponible."""
        features = self._feature_builder.build(indicators, variables or {})

        if self._mode == LearningMode.ONLINE and self._online.is_trained:
            probability = self._online.predict_proba(features)
            source = "online"
        elif self._offline.is_trained:
            probability = self._offline.predict_proba(features)
            source = "offline"
        else:
            probability = 0.5
            source = "none"

        return MLPrediction(
            model_name=f"knowledge_{source}",
            symbol=instrument,
            timeframe=timeframe,
            pattern_name=pattern,
            probability=probability,
            confidence=abs(probability - 0.5) * 2,
            features_used=self._feature_builder.feature_names,
            metadata={"mode": self._mode.value, "source": source},
        )

    async def stats(self) -> dict[str, Any]:
        entries = await self._repo.get_all()
        wins = sum(1 for e in entries if e.outcome == TradeOutcome.WIN)
        losses = sum(1 for e in entries if e.outcome == TradeOutcome.LOSS)
        total = len(entries)
        return {
            "total_entries": total,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total if total else 0.0,
            "mode": self._mode.value,
            "online_samples": self._online.samples_seen,
            "offline_trained": self._offline.is_trained,
            "offline_report": self._offline.last_report,
            "feature_names": self._feature_builder.feature_names,
            "by_pattern": self._group_by_pattern(entries),
            "by_instrument": self._group_by_instrument(entries),
        }

    async def entries(self, **filters: Any) -> list[KnowledgeEntry]:
        return await self._repo.list(**filters)

    @staticmethod
    def _group_by_pattern(entries: list[KnowledgeEntry]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for entry in entries:
            bucket = result.setdefault(entry.pattern, {"count": 0, "wins": 0, "pnl": 0.0})
            bucket["count"] += 1
            bucket["wins"] += entry.is_win
            bucket["pnl"] += entry.pnl
        return result

    @staticmethod
    def _group_by_instrument(entries: list[KnowledgeEntry]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for entry in entries:
            bucket = result.setdefault(entry.instrument, {"count": 0, "wins": 0, "pnl": 0.0})
            bucket["count"] += 1
            bucket["wins"] += entry.is_win
            bucket["pnl"] += entry.pnl
        return result

    @staticmethod
    def _entry_from_trade(trade: Trade | dict[str, Any]) -> KnowledgeEntry:
        if isinstance(trade, Trade):
            pnl = trade.pnl
            direction = trade.direction.value
            tp = trade.take_profit
            sl = trade.stop_loss
            duration = trade.duration or 0.0
            rr = trade.risk_reward_ratio or 0.0
            instrument = trade.symbol
            timeframe = trade.timeframe
            pattern = trade.pattern_name or "unknown"
            score = trade.score
            entry_time = trade.entry_time
            exit_time = trade.exit_time
            metadata = trade.metadata
            drawdown = float(metadata.get("max_adverse_excursion", 0.0) or 0.0)
        else:
            pnl = float(trade.get("pnl", 0.0))
            direction = str(trade.get("direction", "LONG"))
            tp = trade.get("take_profit")
            sl = trade.get("stop_loss")
            entry_time = trade.get("entry_time") or datetime.utcnow()
            exit_time = trade.get("exit_time") or datetime.utcnow()
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time)
            if isinstance(exit_time, str):
                exit_time = datetime.fromisoformat(exit_time)
            duration = (exit_time - entry_time).total_seconds()
            risk = abs(float(trade.get("entry_price", 0.0)) - float(sl)) if sl else 0.0
            reward = abs(float(tp) - float(trade.get("entry_price", 0.0))) if tp else 0.0
            rr = reward / risk if risk > 0 else 0.0
            instrument = str(trade.get("symbol", trade.get("instrument", "UNKNOWN")))
            timeframe = str(trade.get("timeframe", ""))
            pattern = str(trade.get("pattern_name", trade.get("pattern", "unknown")))
            score = float(trade.get("score", 0.0))
            drawdown = float(trade.get("drawdown", 0.0))

        if pnl > 0:
            outcome = TradeOutcome.WIN
        elif pnl < 0:
            outcome = TradeOutcome.LOSS
        else:
            outcome = TradeOutcome.BREAKEVEN

        pnl_pct = pnl / float(trade.get("entry_price", 0.0)) if isinstance(trade, dict) and trade.get("entry_price") else 0.0

        return KnowledgeEntry(
            instrument=instrument,
            timeframe=timeframe,
            pattern=pattern,
            direction=direction,
            outcome=outcome,
            pnl=pnl,
            pnl_pct=pnl_pct,
            drawdown=drawdown,
            take_profit=tp,
            stop_loss=sl,
            risk_reward=rr,
            duration_seconds=duration,
            score=score,
            entry_time=entry_time,
            exit_time=exit_time,
        )

    @staticmethod
    def _trade_from_event(event: Event) -> Optional[Trade]:
        data = event.data or {}
        try:
            trade = Trade(**data)
            return trade
        except Exception as e:  # noqa: BLE001
            logger.warning(f"No se pudo construir Trade desde evento: {e}")
            return None
