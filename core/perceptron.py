"""Entrenamiento del perceptrón simple escalón con la regla del perceptrón.

El escalón no es derivable, así que no se entrena con backprop sino con
Δw = η (y − ŷ) x. Por lotes, sumando sobre las muestras del lote:

    ΔW = η Xᵀ (y − ŷ)        Δb = η Σ (y − ŷ)

Es exactamente lo que calcula Dense.backward_z con delta = ŷ − y seguido de un
paso de GD (θ ← θ − η · grad), así que se reusan la capa y el optimizador sin
tocar el backprop general. Con batch_size=1 es la regla clásica muestra a muestra.
"""

import time
from collections.abc import Sequence

import numpy as np

from core.activations import Step
from core.callbacks import Callback
from core.history import History
from core.network import Network
from core.optimizers import GD


def error_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fracción de muestras mal clasificadas. Ambos (n_muestras, 1) con valores ±1."""
    return float(np.mean(y_true != y_pred))


def fit_perceptron(
    network: Network,
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    epochs: int,
    batch_size: int | None = None,
    callbacks: Sequence[Callback] = (),
    rng: np.random.Generator | None = None,
) -> History:
    """Entrena un perceptrón simple escalón y devuelve el historial por época.

    network: una sola capa, Network([n, 1], output_activation="step").
    X: (n_muestras, n); y: (n_muestras, 1) con valores en {−1, 1}.
    batch_size: None = lote completo, 1 = regla clásica muestra a muestra,
    k = lotes de k barajados cada época.

    En el historial, train_loss es la tasa de error (fracción mal clasificada)
    sobre todo X al terminar la época: 0 significa que clasifica todo bien.
    Una vez en 0 los pesos dejan de cambiar, porque y − ŷ se anula.
    """
    if len(network.layers) != 1 or not isinstance(network.layers[0].activation, Step):
        raise ValueError(
            "fit_perceptron es para un perceptrón simple escalón: una sola capa con salida step."
        )
    n = X.shape[0]
    if y.shape != (n, 1):
        raise ValueError(f"y tiene que tener forma ({n}, 1), llegó {y.shape}")
    if not np.all(np.isin(y, (-1.0, 1.0))):
        raise ValueError("Con salida step las etiquetas tienen que ser −1 o 1.")
    if epochs < 1:
        raise ValueError(f"epochs tiene que ser >= 1, llegó {epochs}")
    if batch_size is not None and batch_size < 1:
        raise ValueError(f"batch_size tiene que ser >= 1 o None, llegó {batch_size}")

    layer = network.layers[0]
    optimizer = GD(lr)
    bs = n if batch_size is None else min(batch_size, n)
    rng = network.rng if rng is None else rng

    history = History()
    for cb in callbacks:
        cb.on_train_begin(network)
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n) if bs < n else np.arange(n)
        for first in range(0, n, bs):
            idx = order[first : first + bs]
            y_pred = network.predict(X[idx])
            # grad = Xᵀ (ŷ − y); el paso de GD resta η·grad, o sea suma η Xᵀ (y − ŷ).
            layer.backward_z(y_pred - y[idx])
            optimizer.step(network.params, network.grads)

        logs = {
            "epoch": epoch,
            "train_loss": error_rate(y, network.predict(X)),
            "elapsed_s": time.perf_counter() - start,
        }
        history.append(logs)
        stop = [cb.on_epoch_end(epoch, logs, network) for cb in callbacks]
        if any(stop):
            break

    for cb in callbacks:
        cb.on_train_end(network)
    return history
