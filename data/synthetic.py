"""Datasets sintéticos para validar el motor (F08).

AND y XOR con las entradas y salidas del enunciado (±1), las mismas que usa
tests/test_validation.py, y muestras de funciones y = f(x) de una variable.
"""

from collections.abc import Callable

import numpy as np

from data.loaders import Dataset

# Las cuatro esquinas del cuadrado [−1, 1]², en el orden del enunciado.
_X_LOGIC = np.array([[-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [1.0, 1.0]])


def _logic_dataset(name: str, y: list[float]) -> Dataset:
    return Dataset(
        X=_X_LOGIC.copy(),
        y=np.array(y, dtype=np.float64).reshape(-1, 1),
        feature_names=["x1", "x2"],
        target_name="y",
        meta={"synthetic": name},
    )


def and_dataset() -> Dataset:
    """AND lógico con ±1: X (4, 2), y (4, 1); y = 1 solo si x1 = x2 = 1."""
    return _logic_dataset("and", [-1.0, -1.0, -1.0, 1.0])


def xor_dataset() -> Dataset:
    """XOR lógico con ±1: X (4, 2), y (4, 1); y = 1 si x1 ≠ x2."""
    return _logic_dataset("xor", [1.0, 1.0, -1.0, -1.0])


def line_samples(
    f: Callable[[np.ndarray], np.ndarray],
    n: int,
    lo: float,
    hi: float,
    rng: np.random.Generator,
) -> Dataset:
    """n muestras x ~ U(lo, hi) con y = f(x): X (n, 1), y (n, 1). P. ej. f = np.tanh."""
    if not lo < hi:
        raise ValueError(f"Hace falta lo < hi, llegó lo = {lo}, hi = {hi}")
    X = rng.uniform(lo, hi, size=(n, 1))
    return Dataset(
        X=X,
        y=np.asarray(f(X), dtype=np.float64).reshape(n, 1),
        feature_names=["x"],
        target_name="y",
        meta={"synthetic": "line", "n": n, "lo": lo, "hi": hi},
    )
