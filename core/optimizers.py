"""Optimizadores: cómo se convierte un gradiente en un paso sobre los parámetros.

Un optimizador recibe las listas paralelas params y grads de la red
(Network.params y Network.grads) y actualiza cada parámetro in place. No
calcula gradientes ni los modifica: eso es responsabilidad del backward.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np


class Optimizer(Protocol):
    def step(self, params: Sequence[np.ndarray], grads: Sequence[np.ndarray]) -> None:
        """Actualiza cada params[i] in place usando grads[i], de la misma forma."""
        ...


class GD:
    """Descenso por gradiente: θ ← θ − lr · g.

    Si es batch completo, estocástico o mini-batch lo decide fit con el tamaño
    del lote; el paso es el mismo.
    """

    def __init__(self, lr: float = 0.01) -> None:
        if lr <= 0:
            raise ValueError(f"lr tiene que ser positivo, llegó {lr}")
        self.lr = lr

    def step(self, params: Sequence[np.ndarray], grads: Sequence[np.ndarray]) -> None:
        for p, g in zip(params, grads, strict=True):
            # In place: p es la misma matriz que guarda la capa.
            p -= self.lr * g


OPTIMIZERS: Mapping[str, type] = MappingProxyType(
    {
        "gd": GD,
    }
)


def get_optimizer(name: str, **kwargs: Any) -> Optimizer:
    """Construye un optimizador a partir de su nombre, p. ej. get_optimizer("gd", lr=0.1).

    Un nombre o parámetro desconocido es un error, no se ignora.
    """
    try:
        cls = OPTIMIZERS[name]
    except KeyError:
        raise ValueError(
            f"Optimizador desconocido: {name!r}. Opciones: {sorted(OPTIMIZERS)}"
        ) from None
    return cls(**kwargs)
