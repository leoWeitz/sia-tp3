"""Red feed-forward: una secuencia de capas densas.

El perceptrón simple es el caso de una sola capa, Network([n, 1], ...); el
multicapa es el caso general. Convención de formas: la entrada X es
(n_muestras, layer_sizes[0]) y la salida (n_muestras, layer_sizes[-1]).
"""

import time
from collections.abc import Callable, Sequence

import numpy as np

from core.activations import Activation, Softmax, get_activation
from core.callbacks import Callback
from core.history import History
from core.initializers import Initializer, get_initializer
from core.layers import Dense
from core.losses import CategoricalCrossEntropy, Loss
from core.optimizers import Optimizer

# Una métrica recibe (y_true, y_pred), ambos (n_muestras, n_salidas), y devuelve un escalar.
Metric = Callable[[np.ndarray, np.ndarray], float]


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
        # Se guarda para barajar los lotes en fit: toda la aleatoriedad de la
        # red (pesos iniciales y orden de las muestras) sale de la misma semilla.
        self.rng = rng

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

    def evaluate(
        self, X: np.ndarray, y: np.ndarray, loss: Loss, metric: Metric | None = None
    ) -> dict[str, float]:
        """Pérdida (y métrica, si se pasa) sobre X completo, sin entrenar.

        X: (n_muestras, layer_sizes[0]); y: (n_muestras, layer_sizes[-1]).
        Devuelve {"loss": ...} y, si hay métrica, también {"metric": ...}.
        """
        y_pred = self.predict(X)
        result = {"loss": loss.value(y, y_pred)}
        if metric is not None:
            result["metric"] = float(metric(y, y_pred))
        return result

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        optimizer: Optimizer,
        loss: Loss,
        epochs: int,
        batch_size: int | None = None,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
        callbacks: Sequence[Callback] = (),
        metric: Metric | None = None,
        rng: np.random.Generator | None = None,
    ) -> History:
        """Entrena la red y devuelve el historial por época.

        X: (n_muestras, layer_sizes[0]); y: (n_muestras, layer_sizes[-1]).
        batch_size: None = lote completo, 1 = estocástico, k = mini-batch de k
        (el último lote de la época puede ser más chico). Con lotes, el orden
        de las muestras se baraja cada época con rng, o con el de la red si no
        se pasa uno.
        validation: (X_val, y_val) para registrar val_loss y val_metric.

        Las pérdidas de cada época se miden sobre el conjunto completo al
        terminar la época, con los pesos ya actualizados, no como promedio de
        los lotes: así una época es comparable con otra.
        """
        n = X.shape[0]
        if y.shape[0] != n:
            raise ValueError(f"X tiene {n} muestras e y tiene {y.shape[0]}")
        if epochs < 1:
            raise ValueError(f"epochs tiene que ser >= 1, llegó {epochs}")
        if batch_size is not None and batch_size < 1:
            raise ValueError(f"batch_size tiene que ser >= 1 o None, llegó {batch_size}")
        bs = n if batch_size is None else min(batch_size, n)
        rng = self.rng if rng is None else rng

        history = History()
        for cb in callbacks:
            cb.on_train_begin(self)
        start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            # Con lote completo el orden no cambia el gradiente: no se baraja.
            order = rng.permutation(n) if bs < n else np.arange(n)
            for first in range(0, n, bs):
                idx = order[first : first + bs]
                X_b, y_b = X[idx], y[idx]
                self.backward(y_b, self.predict(X_b), loss)
                optimizer.step(self.params, self.grads)

            logs: dict[str, float] = {"epoch": epoch}
            train = self.evaluate(X, y, loss, metric)
            logs["train_loss"] = train["loss"]
            if validation is not None:
                val = self.evaluate(*validation, loss, metric)
                logs["val_loss"] = val["loss"]
            if metric is not None:
                logs["train_metric"] = train["metric"]
                if validation is not None:
                    logs["val_metric"] = val["metric"]
            logs["elapsed_s"] = time.perf_counter() - start
            history.append(logs)

            # Lista, no any() con generador: todos los callbacks ven la época
            # aunque uno ya haya pedido cortar.
            stop = [cb.on_epoch_end(epoch, logs, self) for cb in callbacks]
            if any(stop):
                break

        for cb in callbacks:
            cb.on_train_end(self)
        return history

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
