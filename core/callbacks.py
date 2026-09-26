"""Callbacks de entrenamiento: la única forma de observar o cortar un fit.

fit llama a on_train_begin una vez, a on_epoch_end al final de cada época con
los logs de esa época, y a on_train_end al terminar (por épocas agotadas o por
corte temprano). Si algún on_epoch_end devuelve True, el entrenamiento se corta.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.network import Network


class Callback:
    """Base con métodos vacíos: cada callback redefine solo los que usa."""

    def on_train_begin(self, network: "Network") -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: dict[str, float], network: "Network") -> bool:
        """Devuelve True para cortar el entrenamiento."""
        return False

    def on_train_end(self, network: "Network") -> None:
        pass


class PrintProgress(Callback):
    """Reporta los logs cada `every` épocas, y siempre la primera.

    write es obligatorio: el motor nunca imprime por su cuenta, así que quien
    llama a fit decide adónde va el texto (print, un logger, una lista).
    """

    def __init__(self, every: int, write: Callable[[str], None]) -> None:
        if every < 1:
            raise ValueError(f"every tiene que ser >= 1, llegó {every}")
        self.every = every
        self.write = write

    def on_epoch_end(self, epoch: int, logs: dict[str, float], network: "Network") -> bool:
        if epoch == 1 or epoch % self.every == 0:
            parts = [f"{k}={v:.6g}" for k, v in logs.items() if k != "epoch"]
            self.write(f"epoch {epoch}: " + " ".join(parts))
        return False


class EarlyStopping(Callback):
    """Corta si `monitor` no mejora durante `patience` épocas seguidas.

    mode="min" para pérdidas, "max" para métricas como accuracy. Una mejora
    tiene que superar min_delta. Con restore_best=True, al terminar la red
    vuelve a los pesos de la mejor época, haya cortado o no.
    """

    def __init__(
        self,
        patience: int,
        monitor: str = "val_loss",
        mode: str = "min",
        min_delta: float = 0.0,
        restore_best: bool = True,
    ) -> None:
        if patience < 1:
            raise ValueError(f"patience tiene que ser >= 1, llegó {patience}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode tiene que ser 'min' o 'max', llegó {mode!r}")
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best = restore_best

    def on_train_begin(self, network: "Network") -> None:
        self.best: float = np.inf if self.mode == "min" else -np.inf
        self.best_epoch: int | None = None
        self.stopped_epoch: int | None = None
        self._wait = 0
        self._best_params: list[np.ndarray] | None = None

    def on_epoch_end(self, epoch: int, logs: dict[str, float], network: "Network") -> bool:
        if self.monitor not in logs:
            raise ValueError(
                f"EarlyStopping monitorea {self.monitor!r}, que no está en los logs "
                f"({sorted(logs)}). ¿Falta pasar validation o metric a fit?"
            )
        value = logs[self.monitor]
        # Con NaN (entrenamiento divergido) ninguna comparación da True: no
        # cuenta como mejora y se termina cortando por paciencia.
        if self.mode == "min":
            improved = value < self.best - self.min_delta
        else:
            improved = value > self.best + self.min_delta

        if improved:
            self.best = value
            self.best_epoch = epoch
            self._wait = 0
            if self.restore_best:
                self._best_params = [p.copy() for p in network.params]
            return False

        self._wait += 1
        if self._wait >= self.patience:
            self.stopped_epoch = epoch
            return True
        return False

    def on_train_end(self, network: "Network") -> None:
        if self.restore_best and self._best_params is not None:
            for p, best in zip(network.params, self._best_params, strict=True):
                # In place: la capa tiene que seguir apuntando al mismo array.
                p[...] = best
