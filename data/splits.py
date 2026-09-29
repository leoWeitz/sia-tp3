"""Particiones (holdout, k-fold, estratificadas) y preparación de cada fold.

Todas las particiones devuelven ÍNDICES ordenados (int64), nunca copias de los
datos. Toda aleatoriedad sale del np.random.Generator que se recibe.
prepare_fold es el único camino que usan runner y experimentos para pasar de
un Dataset a arrays listos para Network.fit: ajusta la normalización solo con
idx_train (Constitución C5, 04-matematica §8).
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from data.loaders import Dataset
from data.preprocess import TargetScaler, get_scaler, one_hot

Split = tuple[np.ndarray, np.ndarray]

TARGET_ENCODINGS = ("none", "onehot", "pm1_onehot", "scale_to_output")


def _check_ratio(ratio: float) -> None:
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"ratio (fracción de train) tiene que estar en (0, 1), llegó {ratio}")


def _check_k(n: int, k: int) -> None:
    if not 2 <= k <= n:
        raise ValueError(f"k tiene que estar entre 2 y n = {n}, llegó {k}")


def _labels(y: np.ndarray) -> np.ndarray:
    """Etiquetas (n,) desde y (n,) o (n, 1)."""
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1:
        raise ValueError(f"Para estratificar hace falta y (n,) o (n, 1), llegó {y.shape}")
    return y


def _complement(idx: np.ndarray, n: int) -> np.ndarray:
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    return np.flatnonzero(mask)


def holdout(n: int, ratio: float, rng: np.random.Generator) -> Split:
    """Partición aleatoria en train (round(ratio·n) muestras) y validación."""
    _check_ratio(ratio)
    n_train = round(ratio * n)
    if not 0 < n_train < n:
        raise ValueError(f"Con n = {n} y ratio = {ratio} una de las partes queda vacía")
    perm = rng.permutation(n)
    idx_train = np.sort(perm[:n_train])
    return idx_train, _complement(idx_train, n)


def stratified_holdout(y: np.ndarray, ratio: float, rng: np.random.Generator) -> Split:
    """Holdout que mantiene la proporción de cada clase de y en train y validación.

    Cada clase aporta a train ratio·n_c muestras redondeadas por mayor resto
    (así el total es exactamente round(ratio·n) y cada clase queda a ±1 de su
    proporción global). y: etiquetas (n,) o (n, 1); para targets continuos usar
    stratify_labels antes.
    """
    _check_ratio(ratio)
    y = _labels(y)
    n = len(y)
    classes, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    ideal = ratio * counts
    n_per_class = np.floor(ideal).astype(np.int64)
    remaining = round(ratio * n) - n_per_class.sum()
    order = np.argsort(-(ideal - n_per_class), kind="stable")
    n_per_class[order[:remaining]] += 1
    if not 0 < n_per_class.sum() < n:
        raise ValueError(f"Con n = {n} y ratio = {ratio} una de las partes queda vacía")

    chosen = [
        rng.permutation(np.flatnonzero(inverse == c))[:n_c]
        for c, n_c in zip(range(len(classes)), n_per_class, strict=True)
    ]
    idx_train = np.sort(np.concatenate(chosen))
    return idx_train, _complement(idx_train, n)


def _folds_from_assignment(fold_of: np.ndarray, k: int) -> list[Split]:
    n = len(fold_of)
    folds = []
    for f in range(k):
        idx_val = np.flatnonzero(fold_of == f)
        folds.append((_complement(idx_val, n), idx_val))
    return folds


def kfold(n: int, k: int, rng: np.random.Generator) -> list[Split]:
    """k particiones (idx_train, idx_val); los idx_val son disjuntos y cubren todo.

    Los tamaños de validación difieren a lo sumo en 1.
    """
    _check_k(n, k)
    fold_of = np.empty(n, dtype=np.int64)
    fold_of[rng.permutation(n)] = np.arange(n) % k
    return _folds_from_assignment(fold_of, k)


def stratified_kfold(y: np.ndarray, k: int, rng: np.random.Generator) -> list[Split]:
    """k-fold que mantiene la proporción de cada clase en cada fold.

    Se ordenan las muestras por clase (mezcladas dentro de cada clase) y se
    reparten en ronda: cada fold recibe floor o ceil de n_c/k de cada clase y los
    tamaños de validación difieren a lo sumo en 1.
    """
    y = _labels(y)
    n = len(y)
    _check_k(n, k)
    _, inverse = np.unique(y, return_inverse=True)
    shuffled = rng.permutation(n)
    order = shuffled[np.argsort(inverse[shuffled], kind="stable")]
    fold_of = np.empty(n, dtype=np.int64)
    fold_of[order] = np.arange(n) % k
    return _folds_from_assignment(fold_of, k)


def stratify_labels(y: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Discretiza un target continuo (n,) o (n, 1) en bins por cuantiles → etiquetas (n,).

    Sirve para estratificar por la probabilidad de BigModel. Con muchos empates
    (p. ej. muchas probabilidades iguales a 0) los bordes repetidos se fusionan y
    quedan menos de n_bins bins.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins tiene que ser ≥ 1, llegó {n_bins}")
    y = _labels(y).astype(np.float64)
    edges = np.unique(np.quantile(y, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.searchsorted(edges, y, side="right").astype(np.int64)


def prepare_fold(
    dataset: Dataset,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    normalize: str = "minmax",
    target_encoding: str = "none",
    n_classes: int | None = None,
    out_range: tuple[float, float] | None = None,
    scaler_params: Mapping[str, Any] | None = None,
    target_in_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Arma (X_tr, y_tr, X_val, y_val, fitted) listos para Network.fit.

    La normalización (get_scaler(normalize, **scaler_params)) y el TargetScaler
    se ajustan SOLO con idx_train. X_* son (n, n_features) e y_* son (n, k):
    target_encoding
      "none"            → y como float (n, 1) (o (n, k) si ya venía así)
      "onehot"          → one_hot 0/1 con n_classes columnas
      "pm1_onehot"      → one_hot −1/+1 (salida tanh)
      "scale_to_output" → TargetScaler(out_range, target_in_range) ajustado con train
    n_classes: None = max(y) + 1 sobre todo el dataset (solo define la cantidad
    de columnas, no usa estadísticos de validación).
    fitted: {"scaler", "target_scaler" (o None), "n_classes" (o None)}; se usa
    para transformar test e invertir predicciones.
    """
    if target_encoding not in TARGET_ENCODINGS:
        raise ValueError(
            f"target_encoding desconocido: {target_encoding!r}. Opciones: {list(TARGET_ENCODINGS)}"
        )
    idx_train, idx_val = np.asarray(idx_train), np.asarray(idx_val)
    if np.intersect1d(idx_train, idx_val).size:
        raise ValueError("idx_train e idx_val se solapan")

    scaler = get_scaler(normalize, **(scaler_params or {})).fit(dataset.X[idx_train])
    X_tr = scaler.transform(dataset.X[idx_train])
    X_val = scaler.transform(dataset.X[idx_val])

    y = np.asarray(dataset.y)
    y_tr_raw, y_val_raw = y[idx_train], y[idx_val]
    target_scaler = None
    if target_encoding in ("onehot", "pm1_onehot"):
        if n_classes is None:
            n_classes = int(y.max()) + 1
        neg = -1.0 if target_encoding == "pm1_onehot" else 0.0
        y_tr, y_val = one_hot(y_tr_raw, n_classes, neg=neg), one_hot(y_val_raw, n_classes, neg=neg)
    else:
        y_tr = y_tr_raw.astype(np.float64).reshape(len(idx_train), -1)
        y_val = y_val_raw.astype(np.float64).reshape(len(idx_val), -1)
        if target_encoding == "scale_to_output":
            if out_range is None:
                raise ValueError("target_encoding='scale_to_output' necesita out_range")
            target_scaler = TargetScaler(out_range, in_range=target_in_range).fit(y_tr)
            y_tr, y_val = target_scaler.transform(y_tr), target_scaler.transform(y_val)

    fitted = {
        "scaler": scaler,
        "target_scaler": target_scaler,
        "n_classes": n_classes if target_encoding in ("onehot", "pm1_onehot") else None,
    }
    return X_tr, y_tr, X_val, y_val, fitted
