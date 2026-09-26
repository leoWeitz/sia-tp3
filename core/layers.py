"""Capa densa (fully connected).

Convención de formas: la entrada x es (n_muestras, n_in) y la salida es
(n_muestras, n_out). Los pesos son W (n_in, n_out) y el bias b (1, n_out),
que se suma a cada fila por broadcasting; el dataset nunca lleva columna de unos.
"""

import numpy as np

from core.activations import Activation
from core.initializers import Initializer


class Dense:
    """a = activation(x @ W + b).

    forward guarda x y z = x @ W + b porque el backward los necesita: la
    derivada de la activación se evalúa en z, y el gradiente de W usa x.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        activation: Activation,
        initializer: Initializer,
        rng: np.random.Generator,
    ) -> None:
        if n_in < 1 or n_out < 1:
            raise ValueError(f"Tamaños de capa inválidos: n_in={n_in}, n_out={n_out}")
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.W: np.ndarray = initializer(n_in, n_out, rng)
        self.b: np.ndarray = np.zeros((1, n_out))
        # Cache del último forward, para el backward.
        self.x: np.ndarray | None = None
        self.z: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (n_muestras, n_in). Devuelve la activación, (n_muestras, n_out)."""
        if x.ndim != 2 or x.shape[1] != self.n_in:
            raise ValueError(f"Se esperaba entrada (n_muestras, {self.n_in}), llegó {x.shape}")
        self.x = x
        self.z = x @ self.W + self.b
        return self.activation.forward(self.z)

    @property
    def n_params(self) -> int:
        """Cantidad de parámetros entrenables: n_in * n_out pesos más n_out bias."""
        return self.W.size + self.b.size
