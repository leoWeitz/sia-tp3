"""Red feed-forward: una secuencia de capas densas.

El perceptrón simple es el caso de una sola capa, Network([n, 1], ...); el
multicapa es el caso general. Convención de formas: la entrada X es
(n_muestras, layer_sizes[0]) y la salida (n_muestras, layer_sizes[-1]).
"""

from collections.abc import Sequence

import numpy as np

from core.activations import Activation, Softmax, get_activation
from core.initializers import Initializer, get_initializer
from core.layers import Dense
from core.losses import CategoricalCrossEntropy, Loss


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

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray, loss: Loss) -> None:
        """Calcula grad_W y grad_b de todas las capas a partir del último predict.

        y_true e y_pred: (n_muestras, layer_sizes[-1]); y_pred tiene que ser la
        salida del predict inmediatamente anterior, que dejó x y z en cada capa.
        """
        output = self.layers[-1]
        if isinstance(output.activation, Softmax):
            # Par acoplado: dL/dz = (ŷ − y) / n directamente, sin la jacobiana
            # de Softmax. Ver CategoricalCrossEntropy.softmax_delta.
            if not isinstance(loss, CategoricalCrossEntropy):
                raise ValueError(
                    "Una salida Softmax solo se puede entrenar con CategoricalCrossEntropy, "
                    f"llegó {type(loss).__name__}."
                )
            grad = output.backward_z(loss.softmax_delta(y_true, y_pred))
        else:
            grad = output.backward(loss.grad(y_true, y_pred))
        for layer in reversed(self.layers[:-1]):
            grad = layer.backward(grad)

    @property
    def params(self) -> list[np.ndarray]:
        """[W1, b1, W2, b2, ...]. Referencias: el optimizador los actualiza in place."""
        return [p for layer in self.layers for p in layer.params]

    @property
    def grads(self) -> list[np.ndarray]:
        """[grad_W1, grad_b1, ...], en el mismo orden que params."""
        return [g for layer in self.layers for g in layer.grads]

    @property
    def n_params(self) -> int:
        """Cantidad total de parámetros entrenables."""
        return sum(layer.n_params for layer in self.layers)


def _resolve_activation(activation: str | Activation) -> Activation:
    return get_activation(activation) if isinstance(activation, str) else activation
