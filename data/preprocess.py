"""Normalización de features, escalado del target y one-hot.

04-matematica §8. Todos los scalers se ajustan con fit(X_train) y después
transforman train, validación y test con esos mismos estadísticos (Constitución
C5): nunca se ajustan con validación ni test. El único camino que usan runner y
experimentos es data.splits.prepare_fold.

Convención de formas: X es (n_muestras, n_features) y la salida de transform e
inverse_transform tiene la misma forma. Ningún método modifica X in place.
"""

import warnings
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np


def _check_range(feature_range: tuple[float, float], name: str) -> tuple[float, float]:
    a, b = (float(v) for v in feature_range)
    if not a < b:
        raise ValueError(f"{name} tiene que ser (a, b) con a < b, llegó {feature_range}")
    return a, b


class _Scaler:
    """Base: fit / transform / inverse_transform / state_dict / from_state_dict."""

    kind: str = ""
    _state_keys: tuple[str, ...] = ()

    def fit(self, X: np.ndarray) -> "_Scaler":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _check_fitted(self) -> None:
        if any(getattr(self, k) is None for k in self._state_keys):
            raise RuntimeError(f"{type(self).__name__}: hay que llamar a fit antes de transform")

    def _params(self) -> dict[str, Any]:
        return {}

    def state_dict(self) -> dict[str, Any]:
        """Parámetros + estadísticos ajustados, serializable con np.savez / JSON."""
        self._check_fitted()
        state: dict[str, Any] = {"kind": self.kind, **self._params()}
        for k in self._state_keys:
            state[k] = np.array(getattr(self, k))
        return state

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "_Scaler":
        state = dict(state)
        state.pop("kind", None)
        stats = {k: np.asarray(state.pop(k), dtype=np.float64) for k in cls._state_keys}
        scaler = cls(**state)
        for k, v in stats.items():
            setattr(scaler, k, v)
        return scaler


class IdentityScaler(_Scaler):
    """No transforma (normalize = "none"). Devuelve copias."""

    kind = "none"

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.array(X, dtype=np.float64)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.array(X, dtype=np.float64)


class MinMaxScaler(_Scaler):
    """x' = a + (x − min)(b − a) / (max − min), por columna (04-matematica §8).

    Una columna constante en train va a a (y se avisa con warnings.warn).
    """

    kind = "minmax"
    _state_keys = ("data_min", "data_max")

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        self.feature_range = _check_range(feature_range, "feature_range")
        self.data_min: np.ndarray | None = None
        self.data_max: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        X = np.asarray(X, dtype=np.float64)
        self.data_min = X.min(axis=0)
        self.data_max = X.max(axis=0)
        constant = np.flatnonzero(self.data_max == self.data_min)
        if constant.size:
            warnings.warn(
                f"MinMaxScaler: columnas constantes en train {constant.tolist()}; "
                f"se llevan a {self.feature_range[0]}",
                stacklevel=2,
            )
        return self

    def _scale(self) -> np.ndarray:
        """(b − a) / (max − min), con 0 en las columnas constantes."""
        a, b = self.feature_range
        data_range = self.data_max - self.data_min
        safe = np.where(data_range > 0, data_range, 1.0)
        return np.where(data_range > 0, (b - a) / safe, 0.0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.feature_range[0] + (np.asarray(X, dtype=np.float64) - self.data_min) * (
            self._scale()
        )

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        a, b = self.feature_range
        return self.data_min + (np.asarray(X, dtype=np.float64) - a) * (
            (self.data_max - self.data_min) / (b - a)
        )

    def _params(self) -> dict[str, Any]:
        return {"feature_range": list(self.feature_range)}


class StandardScaler(_Scaler):
    """Z-score: x' = (x − media) / s, por columna (04-matematica §8).

    s es el desvío poblacional (ddof = 0) de train; una columna con s = 0 va a 0.
    """

    kind = "zscore"
    _state_keys = ("mean", "std")

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, dtype=np.float64)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        safe = np.where(self.std > 0, self.std, 1.0)
        inv = np.where(self.std > 0, 1.0 / safe, 0.0)
        return (np.asarray(X, dtype=np.float64) - self.mean) * inv

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.mean + np.asarray(X, dtype=np.float64) * self.std


class UnitLengthScaler(_Scaler):
    """x' = x / ‖x‖₂ por fila (muestra), sin estado (04-matematica §8).

    Una fila nula queda nula. No es invertible: la norma de cada fila se pierde.
    """

    kind = "unit_length"

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norms > 0, norms, 1.0)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("UnitLengthScaler no es invertible: pierde la norma de la fila")


SCALERS: Mapping[str, type[_Scaler]] = MappingProxyType(
    {
        "none": IdentityScaler,
        "minmax": MinMaxScaler,
        "zscore": StandardScaler,
        "unit_length": UnitLengthScaler,
    }
)


def get_scaler(name: str, **kwargs: Any) -> _Scaler:
    """Construye un scaler sin ajustar por nombre: "none" | "minmax" | "zscore" | "unit_length".

    Un nombre o parámetro desconocido es un error.
    """
    try:
        cls = SCALERS[name]
    except KeyError:
        raise ValueError(f"Scaler desconocido: {name!r}. Opciones: {sorted(SCALERS)}") from None
    return cls(**kwargs)


def scaler_from_state_dict(state: Mapping[str, Any]) -> _Scaler:
    """Reconstruye un scaler ajustado desde su state_dict (usa la clave "kind")."""
    kind = str(np.asarray(state["kind"]))
    if kind not in SCALERS:
        raise ValueError(f"Scaler desconocido: {kind!r}. Opciones: {sorted(SCALERS)}")
    return SCALERS[kind].from_state_dict(state)


class TargetScaler(_Scaler):
    """Lleva ζ a la imagen de la activación de salida y vuelve (04-matematica §8).

    out_range: imagen de la salida, p. ej. (0, 1) para sigmoide o (−1, 1) para
    tanh. in_range: rango de ζ; si es None se toma min/max de cada columna de
    train en fit. Para probabilidades con salida sigmoide, in_range = out_range =
    (0, 1) da la identidad. Acepta y de forma (n,) o (n, k).
    """

    kind = "target"
    _state_keys = ("data_min", "data_max")

    def __init__(
        self,
        out_range: tuple[float, float],
        in_range: tuple[float, float] | None = None,
    ) -> None:
        self.out_range = _check_range(out_range, "out_range")
        self.in_range = None if in_range is None else _check_range(in_range, "in_range")
        self._minmax = MinMaxScaler(self.out_range)
        if self.in_range is not None:
            self._minmax.data_min = np.array(self.in_range[0])
            self._minmax.data_max = np.array(self.in_range[1])

    @property
    def data_min(self) -> np.ndarray | None:
        return self._minmax.data_min

    @data_min.setter
    def data_min(self, value: np.ndarray) -> None:
        self._minmax.data_min = value

    @property
    def data_max(self) -> np.ndarray | None:
        return self._minmax.data_max

    @data_max.setter
    def data_max(self, value: np.ndarray) -> None:
        self._minmax.data_max = value

    def fit(self, y: np.ndarray) -> "TargetScaler":
        if self.in_range is None:
            self._minmax.fit(np.asarray(y, dtype=np.float64).reshape(len(y), -1))
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        return self._minmax.transform(y.reshape(len(y), -1)).reshape(y.shape)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        return self._minmax.inverse_transform(y.reshape(len(y), -1)).reshape(y.shape)

    def _params(self) -> dict[str, Any]:
        return {
            "out_range": list(self.out_range),
            "in_range": None if self.in_range is None else list(self.in_range),
        }


def one_hot(y: np.ndarray, n_classes: int, neg: float = 0.0, pos: float = 1.0) -> np.ndarray:
    """Etiquetas enteras (n,) o (n, 1) en [0, n_classes) → matriz (n, n_classes).

    La columna de la clase vale pos y el resto neg (neg = −1 para salida tanh).
    """
    labels = np.asarray(y).reshape(-1)
    if not np.all(np.equal(np.mod(labels, 1), 0)):
        raise ValueError("one_hot espera etiquetas enteras")
    labels = labels.astype(np.int64)
    if labels.size and (labels.min() < 0 or labels.max() >= n_classes):
        raise ValueError(
            f"Etiquetas fuera de [0, {n_classes}): min {labels.min()}, max {labels.max()}"
        )
    out = np.full((labels.size, n_classes), neg, dtype=np.float64)
    out[np.arange(labels.size), labels] = pos
    return out
