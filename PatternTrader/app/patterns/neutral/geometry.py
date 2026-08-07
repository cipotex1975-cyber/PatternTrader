from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks as _scipy_find_peaks


def fit_line(values: np.ndarray) -> tuple[float, float]:
    """Ajusta una recta por mínimos cuadrados.

    Devuelve ``(pendiente, intercepto)`` de forma que ``y = pendiente * x + intercepto``.
    """
    x = np.arange(len(values), dtype=float)
    return np.polyfit(x, values, 1)


def slope(values: np.ndarray) -> float:
    """Pendiente de la recta de regresión de ``values``."""
    return fit_line(values)[0]


def line_at(slope: float, intercept: float, x: float) -> float:
    """Valor de la recta definida por ``(slope, intercept)`` en ``x``."""
    return slope * x + intercept


def find_peaks(values: np.ndarray, distance: int = 3) -> list[int]:
    idx, _ = _scipy_find_peaks(values, distance=distance)
    return idx.tolist()


def find_troughs(values: np.ndarray, distance: int = 3) -> list[int]:
    idx, _ = _scipy_find_peaks(-values, distance=distance)
    return idx.tolist()
