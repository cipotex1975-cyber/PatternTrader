from __future__ import annotations

from datetime import datetime

import pytest

from app.signals.models import Signal, SignalPriority
from app.telegram.notifier import TelegramNotifier


def make_signal(**overrides):
    data = dict(
        symbol="BTCUSDT",
        timeframe="1h",
        pattern_name="double_top",
        direction="SHORT",
        priority=SignalPriority.CRITICAL,
        entry_price=50000.0,
        stop_loss=51000.0,
        take_profit=48000.0,
        risk_reward_ratio=2.0,
        score=96.0,
        health=90.0,
        ml_probability=0.85,
        reasons=["Pattern: double_top detected", "Score: 96.0/100 (A)"],
        created_at=datetime(2026, 8, 10, 12, 0, 0),
    )
    data.update(overrides)
    return Signal(**data)


def _enabled_notifier() -> TelegramNotifier:
    notifier = TelegramNotifier()
    notifier._enabled = True
    notifier._retry_backoff = 0
    return notifier


def test_format_signal_message_includes_timeframe_and_date():
    message = _enabled_notifier()._format_signal_message(make_signal())
    assert "1h" in message
    assert "2026-08-10 12:00 UTC" in message
    assert "BTCUSDT" in message
    assert "SHORT" in message


def test_format_signal_message_shows_nd_when_ml_unavailable():
    message = _enabled_notifier()._format_signal_message(make_signal(ml_probability=None))
    assert "Probabilidad IA:</b> N/D" in message
    assert "50%" not in message


def test_format_signal_message_shows_percentage_when_ml_available():
    message = _enabled_notifier()._format_signal_message(make_signal(ml_probability=0.85))
    assert "Probabilidad IA:</b> 85%" in message


@pytest.mark.asyncio
async def test_send_signal_disabled_returns_false():
    notifier = TelegramNotifier()
    assert notifier._enabled is False
    assert await notifier.send_signal(make_signal()) is False


@pytest.mark.asyncio
async def test_send_signal_text_success(monkeypatch):
    notifier = _enabled_notifier()
    calls: list[str] = []

    async def fake_send_message(message: str) -> None:
        calls.append(message)

    monkeypatch.setattr(notifier, "_send_message", fake_send_message)
    assert await notifier.send_signal(make_signal()) is True
    assert len(calls) == 1
    assert "Nueva Señal" in calls[0]


@pytest.mark.asyncio
async def test_send_signal_image_falls_back_to_text(monkeypatch):
    notifier = _enabled_notifier()
    text_calls: list[str] = []

    async def fake_send_photo(message, candles, signal, pattern=None) -> None:
        raise RuntimeError("kaleido/chrome unavailable")

    async def fake_send_message(message: str) -> None:
        text_calls.append(message)

    monkeypatch.setattr(notifier, "_send_photo", fake_send_photo)
    monkeypatch.setattr(notifier, "_send_message", fake_send_message)

    assert await notifier.send_signal(make_signal(), candles=[None]) is True
    assert len(text_calls) == 1


@pytest.mark.asyncio
async def test_send_signal_failure_returns_false(monkeypatch):
    notifier = _enabled_notifier()

    async def fake_send_message(message: str) -> None:
        raise RuntimeError("telegram down")

    monkeypatch.setattr(notifier, "_send_message", fake_send_message)
    assert await notifier.send_signal(make_signal()) is False


@pytest.mark.asyncio
async def test_send_photo_posts_chart(monkeypatch):
    notifier = _enabled_notifier()
    posted: dict = {}

    async def fake_post_with_retries(url: str, **kwargs) -> None:
        posted["url"] = url
        posted["kwargs"] = kwargs

    class FakeFigure:
        def to_image(self, format: str) -> bytes:
            return b"PNGDATA"

    class FakeChartGenerator:
        def create_candlestick_chart(self, candles, title="", patterns=None):
            return FakeFigure()

    monkeypatch.setattr(notifier, "_post_with_retries", fake_post_with_retries)
    monkeypatch.setattr(notifier, "_chart_generator", FakeChartGenerator())

    signal = make_signal()
    assert await notifier.send_signal(signal, candles=[None]) is True
    assert "sendPhoto" in posted["url"]
    assert posted["kwargs"]["files"]["photo"][0] == "chart.png"


@pytest.mark.asyncio
async def test_send_photo_saves_chart_file(monkeypatch, tmp_path):
    notifier = _enabled_notifier()
    notifier._chart_save_dir = str(tmp_path)

    async def fake_post_with_retries(url: str, **kwargs) -> None:
        pass

    class FakeFigure:
        def to_image(self, format: str) -> bytes:
            return b"PNGDATA"

    class FakeChartGenerator:
        def create_candlestick_chart(self, candles, title="", patterns=None):
            return FakeFigure()

    monkeypatch.setattr(notifier, "_post_with_retries", fake_post_with_retries)
    monkeypatch.setattr(notifier, "_chart_generator", FakeChartGenerator())

    assert await notifier.send_signal(make_signal(), candles=[None]) is True

    files = list((tmp_path / "BTCUSDT").glob("*.png"))
    assert len(files) == 1
    name = files[0].name
    assert name.startswith("BTCUSDT_1h_double_top_20260810_120000_")
    assert files[0].read_bytes() == b"PNGDATA"


@pytest.mark.asyncio
async def test_send_photo_does_not_save_when_disabled(monkeypatch, tmp_path):
    notifier = _enabled_notifier()
    notifier._chart_save_dir = ""

    async def fake_post_with_retries(url: str, **kwargs) -> None:
        pass

    class FakeFigure:
        def to_image(self, format: str) -> bytes:
            return b"PNGDATA"

    class FakeChartGenerator:
        def create_candlestick_chart(self, candles, title="", patterns=None):
            return FakeFigure()

    monkeypatch.setattr(notifier, "_post_with_retries", fake_post_with_retries)
    monkeypatch.setattr(notifier, "_chart_generator", FakeChartGenerator())

    assert await notifier.send_signal(make_signal(), candles=[None]) is True
    assert not tmp_path.exists() or not any(tmp_path.rglob("*.png"))


def test_save_chart_png_failure_is_swallowed(tmp_path):
    notifier = _enabled_notifier()
    notifier._chart_save_dir = str(tmp_path / "blocked")
    (tmp_path / "blocked").write_text("not a directory")

    notifier._save_chart_png(make_signal(), b"PNGDATA")


@pytest.mark.asyncio
async def test_post_with_retries_retries_and_succeeds(monkeypatch):
    notifier = _enabled_notifier()
    notifier._max_retries = 3
    attempts = {"n": 0}

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url: str, **kwargs) -> FakeResponse:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("network")
            return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", FakeClient)
    await notifier._post_with_retries("https://api.telegram.org/x")
    assert attempts["n"] == 3


@pytest.mark.asyncio
async def test_post_with_retries_raises_after_exhausting(monkeypatch):
    notifier = _enabled_notifier()
    notifier._max_retries = 2
    attempts = {"n": 0}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url: str, **kwargs):
            attempts["n"] += 1
            raise RuntimeError("network")

    monkeypatch.setattr("httpx.AsyncClient", FakeClient)
    with pytest.raises(RuntimeError):
        await notifier._post_with_retries("https://api.telegram.org/x")
    assert attempts["n"] == 3
