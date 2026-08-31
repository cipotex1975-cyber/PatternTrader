from __future__ import annotations

from typing import NamedTuple

import numpy as np


class WalkForwardFold(NamedTuple):
    """Un fold expanding-window: train previo (cronológicamente) más su validación.

    Ambos segmentos ya vienen recortados de las últimas ``forward_periods``
    muestras (anti-leakage de labels, patrón OPCIÓN B de Fase 2).
    """

    fold_index: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int


def build_walk_forward_folds(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    forward_periods: int = 5,
    min_train_size: int = 100,
    seq_context: int = 30,
) -> list[WalkForwardFold]:
    """Genera folds de validación walk-forward con expanding window (FASE 5).

    Reglas (fase5.md, secciones 2 y 4):
    - Nunca se hace shuffle; los folds son contiguos en el tiempo.
    - Cada ``validation`` fold es estrictamente posterior a su ``train``.
    - ``train`` crece de forma monótona (expanding window): el fold k entrena
      con ``X[0 : train_end_k]`` y valida con el bloque inmediatamente siguiente.
    - Los labels respetan ``forward_periods``: se recortan las últimas
      ``forward_periods`` filas de cada train_fold y de cada validation_fold
      para que ninguna muestra consuma ``high`` de un segmento posterior.

    Args:
        X: matriz de features 2D del conjunto de selección (TRAIN+VALIDATION
            concatenados en orden cronológico).
        y: labels 1D alineados con ``X``.
        n_splits: número de folds walk-forward (>1).
        forward_periods: ventana de etiquetado hacia delante usada por
            ``create_labels`` (para el recorte anti-leakage).
        min_train_size: mínimo de muestras para el train del primer fold.
        seq_context: contexto requerido por modelos secuenciales
            (``sequence_length - 1``).

    Returns:
        Lista de ``WalkForwardFold`` en orden cronológico.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"X debe ser 2D (features); shape {X.shape}")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("y debe ser 1D y estar alineado en longitud con X")
    if n_splits < 2:
        raise ValueError(f"n_splits debe ser >= 2 para walk-forward; recibido {n_splits}")
    if forward_periods < 1:
        raise ValueError(f"forward_periods debe ser >= 1; recibido {forward_periods}")

    n = len(X)
    if n < min_train_size + n_splits + forward_periods * 2:
        raise ValueError(
            f"Dataset demasiado pequeño ({n} muestras) para {n_splits} folds "
            f"walk-forward con min_train_size={min_train_size} y "
            f"forward_periods={forward_periods}"
        )

    # Bloques de validación que reparten el tramo final del conjunto de selección.
    val_block = (n - min_train_size) // n_splits
    if val_block < 1:
        raise ValueError(
            f"No alcanzan muestras para {n_splits} folds walk-forward "
            f"(val_block={val_block}). Reduce n_splits o min_train_size."
        )

    folds: list[WalkForwardFold] = []
    for k in range(n_splits):
        val_start = min_train_size + k * val_block
        val_end = val_start + val_block if k < n_splits - 1 else n
        train_end = val_start

        # Anti-leakage de labels (OPCIÓN B, Fase 2).
        train_trim_end = train_end - forward_periods
        val_trim_end = val_end - forward_periods
        if train_trim_end - 0 < seq_context:
            raise ValueError(
                f"Fold {k}: train demasiado corto para contexto secuencial "
                f"({train_trim_end} < {seq_context})"
            )

        folds.append(
            WalkForwardFold(
                fold_index=k,
                X_train=X[0:train_trim_end],
                y_train=y[0:train_trim_end],
                X_validation=X[val_start:val_trim_end],
                y_validation=y[val_start:val_trim_end],
                train_start=0,
                train_end=train_trim_end,
                validation_start=val_start,
                validation_end=val_trim_end,
            )
        )

    return folds


def validate_walk_forward_no_future(folds: list[WalkForwardFold]) -> None:
    """Verifica la regla 'nunca futuro' (fase5.md, sección 2).

    Cada validation fold debe ser cronológicamente posterior a su train con
    separación estricta de índices (sin overlap ni futuro).
    """
    for fold in folds:
        if fold.validation_start <= fold.train_end:
            raise AssertionError(
                f"Fold {fold.fold_index}: validation ({fold.validation_start}) no es "
                f"posterior a train ({fold.train_end})"
            )
        if fold.validation_end <= fold.validation_start:
            raise AssertionError(f"Fold {fold.fold_index}: validation vacío o inválido")
