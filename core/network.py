"""Red feed-forward: una secuencia de capas densas.

El perceptrón simple es el caso de una sola capa, Network([n, 1], ...); el
multicapa es el caso general. Convención de formas: la entrada X es
(n_muestras, layer_sizes[0]) y la salida (n_muestras, layer_sizes[-1]).
"""

from collections.abc import Sequence

import numpy as np

from core.activations import Activation, get_activation
from core.initializers import Initializer, get_initializer
from core.layers import Dense


class Network:
    """Red densa con una activación para las capas ocultas y otra para la salida.

    Las activaciones y el inicializador se pueden pasar por nombre ("tanh") o
    como objeto ya construido (Tanh(beta=2.0)) cuando llevan parámetros.
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        *,
        hidden_activation: str | Activation = "tanh",
        output_activation: str | Activation = "identity",
        initializer: str | Initializer = "xavier",
        rng: np.random.Generator,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError(
                f"layer_sizes necesita al menos entrada y salida, llegó {list(layer_sizes)}"
            )
        self.layer_sizes = list(layer_sizes)

        hidden = _resolve_activation(hidden_activation)
        output = _resolve_activation(output_activation)
        init = get_initializer(initializer) if isinstance(initializer, str) else initializer

        n_layers = len(self.layer_sizes) - 1
        self.layers = [
            Dense(
                self.layer_sizes[i],
                self.layer_sizes[i + 1],
                output if i == n_layers - 1 else hidden,
                init,
                rng,
            )
            for i in range(n_layers)
        ]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (n_muestras, layer_sizes[0]). Devuelve (n_muestras, layer_sizes[-1])."""
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    @property
    def n_params(self) -> int:
        """Cantidad total de parámetros entrenables."""
        return sum(layer.n_params for layer in self.layers)


def _resolve_activation(activation: str | Activation) -> Activation:
    return get_activation(activation) if isinstance(activation, str) else activation
