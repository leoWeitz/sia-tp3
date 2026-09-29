"""Guardado y carga de redes entrenables: pesos, arquitectura, optimizador y extras.

Un checkpoint es un único .npz escrito con np.savez, que se lee sin pickle:
- W_<i>, b_<i>: pesos (n_in, n_out) y bias (1, n_out) de cada capa.
- optimizer.<clave>: el state_dict del optimizador (lr, t, hiperparámetros y
  buffers), si se pasó uno.
- array.<j>: cada array que aparezca dentro de `extra` o del estado del rng.
- meta: un JSON embebido con layer_sizes, nombre y parámetros (β) de las
  activaciones y del inicializador, el estado del generador de la red
  (rng.bit_generator.state) y `extra`, con cada array reemplazado por una
  referencia a su array.<j>.

load_checkpoint reconstruye una red entrenable, no solo para predecir: el
optimizador sigue con su estado y el rng de la red sigue en el mismo punto,
así que el orden de los lotes continúa igual que si no se hubiera cortado.
El runner (experiments/runner.py) guarda en `extra` la época, el historial y
el estado de los callbacks para reanudar una corrida.
"""

import inspect
import json
import os
from collections.abc import Callable, Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from core.activations import ACTIVATIONS, get_activation
from core.callbacks import Callback
from core.initializers import INITIALIZERS, get_initializer
from core.network import Network
from core.optimizers import Optimizer, get_optimizer

FORMAT_VERSION = 1

# Clave con la que un array de extra queda referenciado dentro del JSON.
_ARRAY_REF = "__ndarray__"
_OPTIMIZER_PREFIX = "optimizer."


def _spec(obj: Any, registry: Mapping[str, type], what: str) -> dict[str, Any]:
    """{"name", "params"} de una activación o inicializador del registro.

    Los parámetros son los argumentos del constructor que el objeto guarda
    como atributo con el mismo nombre (beta, low, high).
    """
    name = next((n for n, cls in registry.items() if type(obj) is cls), None)
    if name is None:
        raise ValueError(
            f"{what} {type(obj).__name__} no está en el registro ({sorted(registry)}): "
            "no se puede serializar por nombre"
        )
    params = {
        p: getattr(obj, p) for p in inspect.signature(type(obj)).parameters if hasattr(obj, p)
    }
    return {"name": name, "params": params}


def _pack(obj: Any, arrays: dict[str, np.ndarray]) -> Any:
    """Copia obj apta para JSON: cada ndarray va a `arrays` y queda una referencia.

    Acepta dicts con claves str, listas, tuplas (quedan como listas), None,
    bool, int, float, str y escalares de NumPy.
    """
    if isinstance(obj, np.ndarray):
        key = f"array.{len(arrays)}"
        arrays[key] = obj
        return {_ARRAY_REF: key}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Mapping):
        packed = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"Las claves de extra tienen que ser str, llegó {k!r}")
            packed[k] = _pack(v, arrays)
        return packed
    if isinstance(obj, (list, tuple)):
        return [_pack(v, arrays) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(f"No se puede guardar un {type(obj).__name__} en un checkpoint")


def _unpack(obj: Any, arrays: Mapping[str, np.ndarray]) -> Any:
    """Inversa de _pack."""
    if isinstance(obj, dict):
        if set(obj) == {_ARRAY_REF}:
            return arrays[obj[_ARRAY_REF]]
        return {k: _unpack(v, arrays) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unpack(v, arrays) for v in obj]
    return obj


def save_checkpoint(
    path: str | PathLike,
    network: Network,
    optimizer: Optimizer | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Guarda la red (y el optimizador y extra, si se pasan) en un .npz.

    Escribe primero a un archivo temporal y lo renombra: si el proceso se
    corta a mitad de la escritura, el checkpoint anterior queda intacto.
    """
    arrays: dict[str, np.ndarray] = {}
    for i, layer in enumerate(network.layers):
        arrays[f"W_{i}"] = layer.W
        arrays[f"b_{i}"] = layer.b
    if optimizer is not None:
        if not hasattr(optimizer, "state_dict"):
            raise ValueError(f"{type(optimizer).__name__} no tiene state_dict: no se puede guardar")
        for key, value in optimizer.state_dict().items():
            arrays[_OPTIMIZER_PREFIX + key] = np.asarray(value)

    hidden = network.layers[0].activation if len(network.layers) > 1 else None
    initializer = getattr(network, "initializer", None)
    referenced: dict[str, np.ndarray] = {}
    meta = {
        "format_version": FORMAT_VERSION,
        "layer_sizes": list(network.layer_sizes),
        "hidden_activation": (
            None if hidden is None else _spec(hidden, ACTIVATIONS, "La activación")
        ),
        "output_activation": _spec(network.layers[-1].activation, ACTIVATIONS, "La activación"),
        "initializer": (
            None if initializer is None else _spec(initializer, INITIALIZERS, "El inicializador")
        ),
        "has_optimizer": optimizer is not None,
        "rng_state": _pack(network.rng.bit_generator.state, referenced),
        "extra": _pack(dict(extra or {}), referenced),
    }
    arrays.update(referenced)
    arrays["meta"] = np.array(json.dumps(meta))

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    # Con un archivo abierto, np.savez no le agrega ".npz" al nombre.
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def load_checkpoint(path: str | PathLike) -> tuple[Network, Optimizer | None, dict[str, Any]]:
    """Reconstruye (network, optimizer o None, extra) desde un .npz de save_checkpoint.

    La red queda con los mismos pesos, activaciones, inicializador y estado
    del rng; el optimizador, con su tipo, lr, t y buffers.
    """
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    meta = json.loads(str(arrays["meta"]))
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"{path}: formato de checkpoint {meta.get('format_version')!r}, "
            f"se esperaba {FORMAT_VERSION}"
        )

    def build(spec: Mapping[str, Any] | None, factory: Callable[..., Any], default: str) -> Any:
        return default if spec is None else factory(spec["name"], **spec["params"])

    rng_state = _unpack(meta["rng_state"], arrays)
    rng = np.random.Generator(getattr(np.random, rng_state["bit_generator"])())
    network = Network(
        meta["layer_sizes"],
        hidden_activation=build(meta["hidden_activation"], get_activation, "tanh"),
        output_activation=build(meta["output_activation"], get_activation, "identity"),
        initializer=build(meta["initializer"], get_initializer, "xavier"),
        rng=rng,
    )
    for i, layer in enumerate(network.layers):
        layer.W[...] = arrays[f"W_{i}"]
        layer.b[...] = arrays[f"b_{i}"]
    # Después de construir la red: la inicialización consumió números del rng.
    rng.bit_generator.state = rng_state

    optimizer = None
    if meta["has_optimizer"]:
        n = len(_OPTIMIZER_PREFIX)
        state = {k[n:]: v for k, v in arrays.items() if k.startswith(_OPTIMIZER_PREFIX)}
        optimizer = get_optimizer(str(state["kind"]), lr=float(state["lr"]))
        optimizer.load_state_dict(state)

    return network, optimizer, _unpack(meta["extra"], arrays)


class Checkpoint(Callback):
    """Guarda un checkpoint con save_checkpoint cada `every` épocas, siempre en `path`.

    extra(epoch, logs) -> dict arma el `extra` de cada checkpoint (lo que haga
    falta para reanudar); sin extra se guarda {"epoch": epoch}. Tiene que ir
    después de los callbacks cuyo estado se quiere guardar, así el checkpoint
    ve lo que cambiaron en esa época (por ejemplo, el lr de AdaptiveEta).
    """

    def __init__(
        self,
        path: str | PathLike,
        every: int,
        optimizer: Optimizer | None = None,
        extra: Callable[[int, dict[str, float]], Mapping[str, Any]] | None = None,
    ) -> None:
        if every < 1:
            raise ValueError(f"every tiene que ser >= 1, llegó {every}")
        self.path = path
        self.every = every
        self.optimizer = optimizer
        self.extra = extra

    def on_epoch_end(self, epoch: int, logs: dict[str, float], network: Network) -> bool:
        if epoch % self.every == 0:
            extra = self.extra(epoch, logs) if self.extra is not None else {"epoch": epoch}
            save_checkpoint(self.path, network, self.optimizer, extra)
        return False
