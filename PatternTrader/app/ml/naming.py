"""Nomenclatura de artefactos ML por par (símbolo + timeframe).

Vive en su propio módulo porque la convención la necesitan tanto el lado que
guarda (``app.ml.training.compare.save_winner``) como el que sirve
(``app.scoring.engine.ScoringEngine``). Duplicar la construcción del nombre en
ambos lados ya costó un bug: el set de exclusión del fallback genérico se quedaba
sin el sufijo de timeframe y acababa cargando como genérico el modelo que era
por par.

Formato: ``{model_name}_{symbol}_{tf}{ext}`` y, para el scaler,
``{model_name}_{symbol}_{tf}.scaler.json``.

``timeframe`` se canonicaliza con ``normalize_timeframe``, de modo que ``H1`` y
``1h`` producen el mismo nombre. Si es ``None`` o vacío se omite el sufijo
(nomenclatura legacy sin timeframe), que el ScoringEngine ignora por resolución
estricta.
"""

from __future__ import annotations

from app.core.constants.market import normalize_timeframe


def model_artifact_stem(model_name: str, symbol: str, timeframe: str | None) -> str:
    """Prefijo común de artefacto y sidecar: ``{modelo}_{símbolo}_{tf}``."""
    canonical_tf = normalize_timeframe(timeframe) if timeframe else ""
    suffix = f"_{canonical_tf}" if canonical_tf else ""
    return f"{model_name}_{symbol}{suffix}"


def model_artifact_name(model_name: str, symbol: str, timeframe: str | None, extension: str) -> str:
    """Nombre del artefacto del modelo: ``{stem}{ext}``."""
    return f"{model_artifact_stem(model_name, symbol, timeframe)}{extension}"


def sidecar_name(model_name: str, symbol: str, timeframe: str | None) -> str:
    """Nombre del sidecar de metadatos: ``{stem}.meta.json``."""
    return f"{model_artifact_stem(model_name, symbol, timeframe)}.meta.json"


def scaler_name(model_name: str, symbol: str, timeframe: str | None, scaler_stem: str) -> str:
    """Nombre del sidecar del scaler: ``{stem}.{scaler_stem}.json``."""
    return f"{model_artifact_stem(model_name, symbol, timeframe)}.{scaler_stem}.json"
