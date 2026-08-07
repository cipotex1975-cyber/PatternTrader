from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from app.database.base import Base


class Asset(Base):
    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(100))
    exchange = Column(String(50))
    asset_type = Column(String(20))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    candles = relationship("Candle", back_populates="asset")
    patterns = relationship("Pattern", back_populates="asset")


class Candle(Base):
    __tablename__ = "candles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    asset = relationship("Asset", back_populates="candles")

    __table_args__ = ({"extend_existing": True},)


class Indicator(Base):
    __tablename__ = "indicators"

    id = Column(Integer, primary_key=True, autoincrement=True)
    candle_id = Column(Integer, ForeignKey("candles.id"), nullable=False)
    name = Column(String(50), nullable=False)
    value = Column(Float)
    parameters = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class Pattern(Base):
    __tablename__ = "patterns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pattern_uuid = Column(String(36), unique=True, nullable=False, index=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    timeframe = Column(String(10), nullable=False)
    pattern_name = Column(String(50), nullable=False)
    pattern_type = Column(String(20))
    confidence = Column(Float)
    health = Column(Float, default=100.0)
    score = Column(Float, default=0.0)
    entry_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    risk_reward_ratio = Column(Float)
    key_levels = Column(JSON)
    status = Column(String(20), default="DETECTED")
    detected_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime)
    metadata_json = Column(JSON)

    asset = relationship("Asset", back_populates="patterns")
    lifecycle = relationship("Lifecycle", back_populates="pattern", uselist=False)


class Lifecycle(Base):
    __tablename__ = "lifecycles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    lifecycle_uuid = Column(String(36), unique=True, nullable=False, index=True)
    pattern_id = Column(Integer, ForeignKey("patterns.id"), nullable=False)
    current_state = Column(String(20), nullable=False)
    transitions = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    closed_at = Column(DateTime)

    pattern = relationship("Pattern", back_populates="lifecycle")


class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_uuid = Column(String(36), unique=True, nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    pattern_name = Column(String(50))
    direction = Column(String(10))
    priority = Column(String(20))
    status = Column(String(20), default="PENDING")
    entry_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    risk_reward_ratio = Column(Float)
    score = Column(Float)
    health = Column(Float)
    ml_probability = Column(Float)
    reasons = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    sent_at = Column(DateTime)
    expires_at = Column(DateTime)
    metadata_json = Column(JSON)


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_uuid = Column(String(36), unique=True, nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10))
    direction = Column(String(10))
    entry_price = Column(Float, nullable=False)
    entry_time = Column(DateTime, nullable=False)
    exit_price = Column(Float)
    exit_time = Column(DateTime)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    size = Column(Float)
    pnl = Column(Float, default=0.0)
    pnl_pct = Column(Float, default=0.0)
    status = Column(String(20), default="OPEN")
    pattern_name = Column(String(50))
    score = Column(Float)
    metadata_json = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class Backtest(Base):
    __tablename__ = "backtests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100))
    config = Column(JSON)
    metrics = Column(JSON)
    trades = Column(JSON, default=list)
    equity_curve = Column(JSON, default=list)
    trades_count = Column(Integer)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    total_pnl = Column(Float)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    initial_capital = Column(Float)
    final_capital = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(JSON)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10))
    pattern_name = Column(String(50))
    probability = Column(Float)
    confidence = Column(Float)
    features_used = Column(JSON)
    actual_outcome = Column(Boolean)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(JSON)


class MLModel(Base):
    __tablename__ = "ml_models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False)
    model_type = Column(String(20))
    version = Column(String(20))
    path = Column(String(255))
    metrics = Column(JSON)
    is_active = Column(Boolean, default=False)
    trained_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(JSON)


class Metric(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    value = Column(Float)
    tags = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)


class Log(Base):
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    level = Column(String(10), nullable=False)
    message = Column(Text)
    source = Column(String(100))
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(JSON)


class KnowledgeEntry(Base):
    __tablename__ = "knowledge_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entry_uuid = Column(String(36), unique=True, nullable=False, index=True)
    instrument = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    pattern = Column(String(50), nullable=False, index=True)
    direction = Column(String(10), default="LONG")
    variables = Column(JSON, default=dict)
    indicators = Column(JSON, default=dict)
    outcome = Column(String(20), nullable=False)
    pnl = Column(Float, default=0.0)
    pnl_pct = Column(Float, default=0.0)
    drawdown = Column(Float, default=0.0)
    take_profit = Column(Float)
    stop_loss = Column(Float)
    risk_reward = Column(Float, default=0.0)
    duration_seconds = Column(Float, default=0.0)
    score = Column(Float, default=0.0)
    entry_time = Column(DateTime)
    exit_time = Column(DateTime)
    image_path = Column(String(255), default="")
    ml_features = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
