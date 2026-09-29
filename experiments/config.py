"""Configuración de experimentos: defaults, validación, barridos, hash y construcción.

Contrato completo en docs/03-arquitectura.md §4 (y CLAUDE.md §6). Un JSON mínimo
alcanza (run_name, dataset, model.layers, training.epochs): todo lo demás tiene
default acá. Una clave desconocida, un tipo inválido o una combinación
incompatible es un ConfigError cuyo mensaje nombra la clave con notación punto
(p. ej. "training.optimizer.alpha").

Hay dos formas de config resuelta:
- base (load_config, resolve_config): con "seeds" y "sweep", describe todo un
  barrido;
- corrida (expand): una combinación del sweep con "seed" y "fold" fijos y sin
  "seeds". Es la que el runner guarda en results/<run_name>/<run_id>/config.json.
"""

import copy
import hashlib
import inspect
import itertools
import json
import re
from collections.abc import Callable, Mapping
from os import PathLike
from typing import Any, NoReturn

import numpy as np

from core.activations import ACTIVATIONS, Activation, get_activation
from core.augmentation import AUGMENTATIONS, Augmentation, get_augmentation
from core.callbacks import AdaptiveEta, Callback, EarlyStopping
from core.initializers import INITIALIZERS, get_initializer
from core.losses import LOSSES, Loss, get_loss
from core.metrics import METRIC_NAMES, TASKS, Metric, get_metric
from core.network import Network
from core.optimizers import GD, OPTIMIZERS, Optimizer, get_optimizer
from data.loaders import DIGITS_IMAGE_SHAPE, DIGITS_N_PIXELS
from data.preprocess import SCALERS
from data.splits import TARGET_ENCODINGS


class ConfigError(ValueError):
    """Config inválida. El mensaje empieza con la clave en notación punto."""


class _Required:
    """Marca de clave obligatoria (sin default). deepcopy la conserva."""

    def __deepcopy__(self, memo: dict) -> "_Required":
        return self

    def __repr__(self) -> str:
        return "<obligatoria>"


REQUIRED = _Required()

TRAINERS = ("backprop", "perceptron")
FORMATS = ("tabular", "digits")
TASK_CHOICES = ("auto", *TASKS)
SPLIT_KINDS = ("none", "holdout", "kfold")
SYNTHETIC_NAMES = ("and", "xor", "line")
LINE_FUNCTIONS = ("identity", "tanh")
FINAL_MODES = ("retrain", "reuse")
DIGITS_N_CLASSES = 10
ONEHOT_ENCODINGS = ("onehot", "pm1_onehot")
# Monitores cuyo valor se minimiza (EarlyStopping mode="min"); el resto se maximiza.
MINIMIZED = ("loss", "mse", "mae")

# Claves que no cambian lo que aprende una corrida: no entran al hash y no se
# pueden barrer (dos valores distintos caerían en la misma carpeta).
HASH_EXCLUDED = ("run_name", "seed", "seeds", "fold", "sweep", "logging")

_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

_TOP = {
    "run_name": REQUIRED,
    "seeds": [0, 1, 2],
    "trainer": "backprop",
    "dataset": REQUIRED,
    "model": REQUIRED,
    "training": REQUIRED,
    "metrics": [],
    "logging": None,
    "sweep": {},
    "selected_from": None,
    "final": None,
}
_DATASET = {
    "name": None,
    "format": "tabular",
    "path": None,
    "synthetic": None,
    "target": None,
    "features": None,
    "drop": [],
    "categorical": [],
    "na_policy": "error",
    "task": "auto",
    "n_classes": None,
    "normalize": "minmax",
    "feature_range": None,
    "target_encoding": "none",
    "target_in_range": None,
    "split": None,
    "holdout_test": None,
    "test_path": None,
}
_SPLITS = {
    "none": {"kind": "none"},
    "holdout": {"kind": "holdout", "ratio": 0.8, "stratified": True, "seed": None},
    "kfold": {"kind": "kfold", "k": 5, "stratified": True, "seed": None},
}
_HOLDOUT_TEST = {"ratio": 0.2, "seed": 0, "stratified": True, "indices_path": None}
_SYNTHETIC = {
    "and": {"name": "and"},
    "xor": {"name": "xor"},
    "line": {"name": "line", "function": "identity", "n": 50, "low": -1.0, "high": 1.0, "seed": 0},
}
_MODEL = {
    "layers": REQUIRED,
    "hidden_activation": "tanh",
    "output_activation": "identity",
    "beta": 1.0,
    "initializer": "xavier",
    "init_params": {},
}
_TRAINING = {
    "loss": "mse",
    "optimizer": None,
    "batch_size": None,
    "epochs": REQUIRED,
    "l2": 0.0,
    "augmentation": None,
    "early_stopping": None,
    "adaptive_eta": None,
}
_LOGGING = {"every": 10, "save_model": True, "save_predictions": False, "checkpoint_every": 50}
_FINAL = {"mode": "retrain", "epochs": None, "threshold": None}


# --- Chequeos de tipo ---


def _fail(path: str, message: str) -> NoReturn:
    raise ConfigError(f"{path}: {message}")


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_int(value: Any, path: str, minimum: int | None = None, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not _is_int(value) or (minimum is not None and value < minimum):
        expected = "un entero" + (f" >= {minimum}" if minimum is not None else "")
        _fail(path, f"tiene que ser {expected}{' o null' if optional else ''}, llegó {value!r}")


def _check_number(value: Any, path: str, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not _is_number(value):
        _fail(path, f"tiene que ser un número{' o null' if optional else ''}, llegó {value!r}")


def _check_bool(value: Any, path: str) -> None:
    if not isinstance(value, bool):
        _fail(path, f"tiene que ser true o false, llegó {value!r}")


def _check_str(value: Any, path: str, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not value:
        _fail(
            path, f"tiene que ser un texto no vacío{' o null' if optional else ''}, llegó {value!r}"
        )


def _check_choice(value: Any, path: str, choices: Any) -> None:
    if not isinstance(value, str) or value not in choices:
        _fail(path, f"valor desconocido {value!r}. Opciones: {sorted(choices)}")


def _check_str_list(value: Any, path: str, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        _fail(path, f"tiene que ser una lista de textos, llegó {value!r}")


def _check_range(value: Any, path: str, optional: bool = True) -> None:
    if optional and value is None:
        return
    ok = isinstance(value, list) and len(value) == 2 and all(_is_number(v) for v in value)
    if not ok or not value[0] < value[1]:
        _fail(path, f"tiene que ser [a, b] con a < b, llegó {value!r}")


def _section(raw: Any, path: str, defaults: Mapping[str, Any]) -> dict[str, Any]:
    """Copia de raw con los defaults completados; clave desconocida o faltante es error."""
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        _fail(path or "config", f"tiene que ser un objeto, llegó {raw!r}")
    unknown = [k for k in raw if k not in defaults]
    if unknown:
        _fail(_join(path, str(unknown[0])), f"clave desconocida. Opciones: {sorted(defaults)}")
    out: dict[str, Any] = {}
    for key, default in defaults.items():
        if key in raw:
            out[key] = copy.deepcopy(raw[key])
        elif default is REQUIRED:
            _fail(_join(path, key), "es obligatoria")
        else:
            out[key] = copy.deepcopy(default)
    return out


def _signature(cls: type, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    """Parámetros del constructor de cls con su default (REQUIRED si no tiene)."""
    return {
        name: REQUIRED if p.default is inspect.Parameter.empty else p.default
        for name, p in inspect.signature(cls).parameters.items()
        if name not in exclude
    }


def _check_params(
    params: Mapping[str, Any], path: str, cls: type, exclude: tuple[str, ...] = ()
) -> None:
    allowed = _signature(cls, exclude)
    for key in params:
        if key not in allowed:
            _fail(
                _join(path, key),
                f"parámetro desconocido para {cls.__name__}. Opciones: {sorted(allowed)}",
            )


def _try(path: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Llama a fn; un TypeError o ValueError pasa a ser ConfigError con la clave."""
    try:
        return fn(*args, **kwargs)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"{path}: {e}") from None


# --- Secciones ---


def _resolve_split(raw: Any, path: str) -> dict[str, Any]:
    raw = {"kind": "holdout"} if raw is None else raw
    if not isinstance(raw, Mapping):
        _fail(path, f"tiene que ser un objeto, llegó {raw!r}")
    kind = raw.get("kind", "holdout")
    _check_choice(kind, _join(path, "kind"), SPLIT_KINDS)
    split = _section(raw, path, _SPLITS[kind])
    if kind == "holdout":
        _check_number(split["ratio"], _join(path, "ratio"))
        if not 0 < split["ratio"] < 1:
            _fail(_join(path, "ratio"), f"tiene que estar en (0, 1), llegó {split['ratio']}")
    if kind == "kfold":
        _check_int(split["k"], _join(path, "k"), minimum=2)
    if kind != "none":
        _check_bool(split["stratified"], _join(path, "stratified"))
        _check_int(split["seed"], _join(path, "seed"), optional=True)
    return split


def _resolve_synthetic(raw: Any, path: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        _fail(path, f"tiene que ser un objeto con 'name', llegó {raw!r}")
    name = raw.get("name")
    _check_choice(name, _join(path, "name"), SYNTHETIC_NAMES)
    synthetic = _section(raw, path, _SYNTHETIC[name])
    if name == "line":
        _check_choice(synthetic["function"], _join(path, "function"), LINE_FUNCTIONS)
        _check_int(synthetic["n"], _join(path, "n"), minimum=2)
        _check_int(synthetic["seed"], _join(path, "seed"))
        _check_range([synthetic["low"], synthetic["high"]], f"{path}.low/high", optional=False)
    return synthetic


def _resolve_dataset(raw: Any, final_eval: bool) -> dict[str, Any]:
    ds = _section(raw, "dataset", _DATASET)
    _check_str(ds["name"], "dataset.name", optional=True)
    _check_choice(ds["format"], "dataset.format", FORMATS)
    if (ds["path"] is None) == (ds["synthetic"] is None):
        _fail("dataset", "tiene que tener dataset.path o dataset.synthetic (uno solo)")

    defaults_only = ("target", "features", "drop", "categorical", "na_policy", "test_path")
    if ds["synthetic"] is not None:
        ds["synthetic"] = _resolve_synthetic(ds["synthetic"], "dataset.synthetic")
        if ds["format"] != "tabular":
            _fail("dataset.format", "no aplica a un dataset sintético")
        for key in defaults_only:
            if ds[key] != _DATASET[key]:
                _fail(f"dataset.{key}", "no aplica a un dataset sintético")
    else:
        _check_str(ds["path"], "dataset.path")
        if ds["format"] == "tabular":
            _check_str(ds["target"], "dataset.target")
            _check_str_list(ds["features"], "dataset.features", optional=True)
            _check_str_list(ds["drop"], "dataset.drop")
            _check_str_list(ds["categorical"], "dataset.categorical")
            _check_choice(ds["na_policy"], "dataset.na_policy", ("error", "drop_rows"))
        else:
            if ds["target"] not in (None, "label"):
                _fail("dataset.target", "en format 'digits' el target es siempre 'label'")
            for key in ("features", "drop", "categorical", "na_policy"):
                if ds[key] != _DATASET[key]:
                    _fail(f"dataset.{key}", "no aplica a format 'digits'")

    _check_choice(ds["task"], "dataset.task", TASK_CHOICES)
    _check_int(ds["n_classes"], "dataset.n_classes", minimum=2, optional=True)
    _check_choice(ds["normalize"], "dataset.normalize", SCALERS)
    _check_range(ds["feature_range"], "dataset.feature_range")
    if ds["feature_range"] is not None and ds["normalize"] != "minmax":
        _fail("dataset.feature_range", "solo aplica con normalize 'minmax'")
    _check_choice(ds["target_encoding"], "dataset.target_encoding", TARGET_ENCODINGS)
    _check_range(ds["target_in_range"], "dataset.target_in_range")
    if ds["target_in_range"] is not None and ds["target_encoding"] != "scale_to_output":
        _fail("dataset.target_in_range", "solo aplica con target_encoding 'scale_to_output'")
    if ds["target_encoding"] in ONEHOT_ENCODINGS:
        if ds["task"] not in ("auto", "multiclass"):
            _fail("dataset.task", f"con target_encoding {ds['target_encoding']!r} es 'multiclass'")
        if ds["synthetic"] is not None:
            _fail("dataset.target_encoding", "los datasets sintéticos no tienen clases enteras")
    elif ds["n_classes"] is not None:
        _fail("dataset.n_classes", "solo aplica con target_encoding 'onehot' o 'pm1_onehot'")
    elif ds["task"] == "multiclass":
        _fail("dataset.task", "'multiclass' necesita target_encoding 'onehot' o 'pm1_onehot'")

    ds["split"] = _resolve_split(ds["split"], "dataset.split")
    if ds["holdout_test"] is not None:
        test = _section(ds["holdout_test"], "dataset.holdout_test", _HOLDOUT_TEST)
        _check_number(test["ratio"], "dataset.holdout_test.ratio")
        if not 0 < test["ratio"] < 1:
            _fail("dataset.holdout_test.ratio", f"tiene que estar en (0, 1), llegó {test['ratio']}")
        _check_int(test["seed"], "dataset.holdout_test.seed")
        _check_bool(test["stratified"], "dataset.holdout_test.stratified")
        _check_str(test["indices_path"], "dataset.holdout_test.indices_path", optional=True)
        ds["holdout_test"] = test
    _check_str(ds["test_path"], "dataset.test_path", optional=True)
    if ds["test_path"] is not None:
        if not final_eval:
            _fail("dataset.test_path", "solo se acepta con --final-eval (el test no decide nada)")
        if ds["holdout_test"] is not None:
            _fail("dataset", "test_path y holdout_test son dos tests distintos: usar uno")
    return ds


def _activation_name(value: Any, path: str, hidden: bool) -> None:
    _check_choice(value, path, ACTIVATIONS)
    if hidden and value in ("softmax", "step"):
        _fail(path, f"{value!r} no se puede usar en capas ocultas")


def _resolve_model(raw: Any) -> dict[str, Any]:
    model = _section(raw, "model", _MODEL)
    layers = model["layers"]
    if not isinstance(layers, list) or len(layers) < 2:
        _fail("model.layers", f"tiene que ser una lista de al menos 2 tamaños, llegó {layers!r}")
    for i, size in enumerate(layers):
        if i == 0 and size == "auto":
            continue
        _check_int(size, f"model.layers[{i}]", minimum=1)
    _activation_name(model["hidden_activation"], "model.hidden_activation", hidden=True)
    _activation_name(model["output_activation"], "model.output_activation", hidden=False)
    _check_number(model["beta"], "model.beta")
    if not model["beta"] > 0:
        _fail("model.beta", f"tiene que ser positivo, llegó {model['beta']}")
    _check_choice(model["initializer"], "model.initializer", INITIALIZERS)
    if not isinstance(model["init_params"], Mapping):
        _fail("model.init_params", f"tiene que ser un objeto, llegó {model['init_params']!r}")
    _check_params(model["init_params"], "model.init_params", INITIALIZERS[model["initializer"]])
    _try("model.init_params", get_initializer, model["initializer"], **model["init_params"])
    return model


def _resolve_optimizer(raw: Any) -> dict[str, Any]:
    path = "training.optimizer"
    raw = {"kind": "gd"} if raw is None else raw
    if not isinstance(raw, Mapping):
        _fail(path, f"tiene que ser un objeto con 'kind', llegó {raw!r}")
    kind = raw.get("kind", "gd")
    _check_choice(kind, _join(path, "kind"), OPTIMIZERS)
    params = {k: v for k, v in raw.items() if k != "kind"}
    _check_params(params, path, OPTIMIZERS[kind])
    for key, value in params.items():
        _check_number(value, _join(path, key))
    # Solo se completa lr (con el default de la clase): los demás
    # hiperparámetros quedan implícitos, así barrer "kind" no arrastra
    # parámetros que el otro optimizador no acepta.
    params.setdefault("lr", _signature(OPTIMIZERS[kind])["lr"])
    _try(path, get_optimizer, kind, **params)
    return {"kind": kind, **params}


def _resolve_augmentation(raw: Any, fmt: str) -> Any:
    path = "training.augmentation"
    if raw is None:
        return None
    items = raw if isinstance(raw, list) else [raw]
    if not items:
        _fail(path, "una lista de augmentations no puede estar vacía")
    resolved = []
    for i, item in enumerate(items):
        item_path = f"{path}[{i}]" if isinstance(raw, list) else path
        if not isinstance(item, Mapping):
            _fail(item_path, f"tiene que ser un objeto con 'kind', llegó {item!r}")
        kind = item.get("kind")
        _check_choice(kind, _join(item_path, "kind"), AUGMENTATIONS)
        params = {k: copy.deepcopy(v) for k, v in item.items() if k != "kind"}
        _check_params(params, item_path, AUGMENTATIONS[kind])
        if kind == "random_shift" and "image_shape" not in params and fmt == "digits":
            params["image_shape"] = list(DIGITS_IMAGE_SHAPE)
        _try(item_path, get_augmentation, {"kind": kind, **params})
        resolved.append({"kind": kind, **params})
    return resolved if isinstance(raw, list) else resolved[0]


def _resolve_callback(
    raw: Any, path: str, cls: type, exclude: tuple[str, ...] = ()
) -> dict[str, Any] | None:
    if raw is None:
        return None
    section = _section(raw, path, _signature(cls, exclude))
    for key, value in section.items():
        key_path = _join(path, key)
        if key in ("patience", "k", "k_prime"):
            _check_int(value, key_path, minimum=1)
        elif key in ("monitor", "mode"):
            _check_str(value, key_path)
        elif key == "restore_best":
            _check_bool(value, key_path)
        else:
            _check_number(value, key_path)
    return section


def _resolve_training(raw: Any, fmt: str) -> dict[str, Any]:
    training = _section(raw, "training", _TRAINING)
    _check_choice(training["loss"], "training.loss", LOSSES)
    training["optimizer"] = _resolve_optimizer(training["optimizer"])
    _check_int(training["batch_size"], "training.batch_size", minimum=1, optional=True)
    _check_int(training["epochs"], "training.epochs", minimum=1)
    _check_number(training["l2"], "training.l2")
    if training["l2"] < 0:
        _fail("training.l2", f"no puede ser negativo, llegó {training['l2']}")
    training["augmentation"] = _resolve_augmentation(training["augmentation"], fmt)

    es = _resolve_callback(training["early_stopping"], "training.early_stopping", EarlyStopping)
    if es is not None:
        _try("training.early_stopping", EarlyStopping, **es)
    training["early_stopping"] = es
    ae = _resolve_callback(
        training["adaptive_eta"], "training.adaptive_eta", AdaptiveEta, exclude=("optimizer",)
    )
    if ae is not None:
        _try("training.adaptive_eta", AdaptiveEta, GD(), **ae)
    training["adaptive_eta"] = ae
    return training


def _resolve_sweep(raw: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        _fail("sweep", f"tiene que ser un objeto {{clave: [valores]}}, llegó {raw!r}")
    for key, values in raw.items():
        path = f"sweep[{key!r}]"
        if not isinstance(key, str) or not key:
            _fail("sweep", f"clave inválida {key!r}")
        if key.split(".")[0] in HASH_EXCLUDED:
            _fail(path, f"no se puede barrer {key!r}: no cambia el hash de la corrida")
        if not isinstance(values, list) or not values:
            _fail(path, f"tiene que ser una lista no vacía de valores, llegó {values!r}")
        dumped = [json.dumps(v, sort_keys=True) for v in values]
        if len(set(dumped)) != len(dumped):
            _fail(path, f"tiene valores repetidos: {values}")
        parent = key.rpartition(".")[0]
        if parent and not isinstance(_get(config, parent, path), Mapping):
            _fail(path, f"{parent!r} no es un objeto de la config (¿es null?)")
    return copy.deepcopy(dict(raw))


# --- Chequeos entre secciones ---


def static_task(config: Mapping[str, Any]) -> str | None:
    """Tarea del problema si se puede saber sin leer los datos; None si no."""
    ds = config["dataset"]
    if ds["task"] != "auto":
        return ds["task"]
    if ds["target_encoding"] in ONEHOT_ENCODINGS:
        return "multiclass"
    if ds["synthetic"] is not None:
        return "regression" if ds["synthetic"]["name"] == "line" else "binary"
    return None


def _expected_widths(config: Mapping[str, Any]) -> tuple[int | None, int | None]:
    """(features, columnas del target codificado), None donde depende de los datos."""
    ds = config["dataset"]
    if ds["synthetic"] is not None:
        n_in = 1 if ds["synthetic"]["name"] == "line" else 2
    else:
        n_in = DIGITS_N_PIXELS if ds["format"] == "digits" else None
    if ds["target_encoding"] in ONEHOT_ENCODINGS:
        n_out = ds["n_classes"]
        if n_out is None and ds["format"] == "digits":
            n_out = DIGITS_N_CLASSES
    else:
        # Un target tabular es una sola columna; el label de dígitos, un entero.
        n_out = 1
    return n_in, n_out


def _log_keys(config: Mapping[str, Any]) -> set[str]:
    """Claves de los logs de cada época que puede monitorear un callback."""
    if config["trainer"] == "perceptron":
        return {"train_loss"}
    names = ["loss", *config["metrics"]]
    keys = {f"train_{m}" for m in names}
    if config["dataset"]["split"]["kind"] != "none":
        keys |= {f"val_{m}" for m in names}
    return keys


def _check_monitor(config: Mapping[str, Any], section: str, check_mode: bool) -> None:
    callback = config["training"][section]
    if callback is None:
        return
    path = f"training.{section}"
    monitor = callback["monitor"]
    available = _log_keys(config)
    if monitor not in available:
        hint = (
            " (val_* necesita dataset.split distinto de 'none')"
            if monitor.startswith("val_")
            else ""
        )
        _fail(
            f"{path}.monitor",
            f"{monitor!r} no está en los logs{hint}. Opciones: {sorted(available)}",
        )
    if check_mode:
        minimize = monitor.split("_", 1)[1] in MINIMIZED
        if minimize != (callback["mode"] == "min"):
            expected = "min" if minimize else "max"
            _fail(
                f"{path}.mode",
                f"{monitor!r} se {'minimiza' if minimize else 'maximiza'}: usar mode {expected!r}",
            )


def _check_consistency(config: dict[str, Any]) -> None:
    ds, model, training = config["dataset"], config["model"], config["training"]
    output, loss = model["output_activation"], training["loss"]
    if output == "softmax" and loss != "categorical_crossentropy":
        _fail("training.loss", "una salida softmax solo se entrena con 'categorical_crossentropy'")
    if loss == "binary_crossentropy" and output != "sigmoid":
        _fail("training.loss", "'binary_crossentropy' necesita salida 'sigmoid'")
    if output == "softmax" and ds["target_encoding"] != "onehot":
        _fail("dataset.target_encoding", "una salida softmax necesita target_encoding 'onehot'")
    if ds["target_encoding"] == "scale_to_output" and output not in ("tanh", "sigmoid"):
        _fail("dataset.target_encoding", "'scale_to_output' necesita salida 'tanh' o 'sigmoid'")

    if config["trainer"] == "perceptron":
        if output != "step" or len(model["layers"]) != 2 or model["layers"][-1] != 1:
            _fail(
                "trainer",
                "'perceptron' es para un perceptrón simple escalón: layers [n, 1] y salida 'step'",
            )
        if training["optimizer"] != {"kind": "gd", "lr": training["optimizer"]["lr"]}:
            _fail(
                "training.optimizer",
                "el trainer 'perceptron' usa la regla del perceptrón: solo {'kind': 'gd', 'lr': η}",
            )
        for key in ("augmentation", "adaptive_eta"):
            if training[key] is not None:
                _fail(f"training.{key}", "no aplica al trainer 'perceptron'")
        if training["l2"] != 0:
            _fail("training.l2", "no aplica al trainer 'perceptron'")
        if ds["target_encoding"] != "none":
            _fail("dataset.target_encoding", "el trainer 'perceptron' usa etiquetas ±1 ('none')")
    elif output == "step":
        _fail("model.output_activation", "'step' no es derivable: usar trainer 'perceptron'")

    n_in, n_out = _expected_widths(config)
    layers = model["layers"]
    if n_in is not None and layers[0] not in ("auto", n_in):
        _fail("model.layers[0]", f"vale {layers[0]} y el dataset tiene {n_in} features")
    if n_out is not None and layers[-1] != n_out:
        _fail(
            "model.layers[-1]", f"vale {layers[-1]} y el target codificado tiene {n_out} columna(s)"
        )
    if ds["target_encoding"] in ONEHOT_ENCODINGS and layers[-1] < 2:
        _fail("model.layers[-1]", "one-hot necesita al menos 2 salidas")

    task = static_task(config)
    for i, name in enumerate(config["metrics"]):
        _check_choice(name, f"metrics[{i}]", METRIC_NAMES)
        if task is not None:
            _try(f"metrics[{i}]", get_metric, name, task=task)
    if len(set(config["metrics"])) != len(config["metrics"]):
        _fail("metrics", f"tiene nombres repetidos: {config['metrics']}")
    _check_monitor(config, "early_stopping", check_mode=True)
    _check_monitor(config, "adaptive_eta", check_mode=False)


# --- Resolución ---


def _resolve(raw: Any, *, final_eval: bool, run: bool) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        _fail("config", f"tiene que ser un objeto JSON, llegó {type(raw).__name__}")
    raw = dict(raw)
    if run:
        top = {"run_name": REQUIRED, "seed": REQUIRED, "fold": None}
        top.update({k: v for k, v in _TOP.items() if k not in top and k != "seeds"})
    else:
        top = dict(_TOP)
        if "fold" in raw:
            _fail("fold", "no va en la config: sale de expandir un kfold (usar --only fold=k)")
        if "seed" in raw:
            if "seeds" in raw:
                _fail("seed", "usar seed o seeds, no las dos")
            raw["seeds"] = [raw.pop("seed")]
    config = _section(raw, "", top)

    _check_str(config["run_name"], "run_name")
    if not _RUN_NAME.match(config["run_name"]):
        _fail("run_name", f"solo letras, números, '_', '.' y '-', llegó {config['run_name']!r}")
    if run:
        _check_int(config["seed"], "seed", minimum=0)
    else:
        seeds = config["seeds"]
        if not isinstance(seeds, list) or not seeds:
            _fail("seeds", f"tiene que ser una lista no vacía de enteros, llegó {seeds!r}")
        for i, seed in enumerate(seeds):
            _check_int(seed, f"seeds[{i}]", minimum=0)
        if len(set(seeds)) != len(seeds):
            _fail("seeds", f"tiene semillas repetidas: {seeds}")
    _check_choice(config["trainer"], "trainer", TRAINERS)

    config["dataset"] = _resolve_dataset(config["dataset"], final_eval)
    config["model"] = _resolve_model(config["model"])
    config["training"] = _resolve_training(config["training"], config["dataset"]["format"])
    if not isinstance(config["metrics"], list):
        _fail("metrics", f"tiene que ser una lista de nombres, llegó {config['metrics']!r}")
    config["logging"] = _section(config["logging"], "logging", _LOGGING)
    logging = config["logging"]
    _check_int(logging["every"], "logging.every", minimum=1)
    _check_bool(logging["save_model"], "logging.save_model")
    _check_bool(logging["save_predictions"], "logging.save_predictions")
    _check_int(logging["checkpoint_every"], "logging.checkpoint_every", minimum=0, optional=True)
    config["sweep"] = _resolve_sweep(config["sweep"], config)

    if final_eval:
        ds = config["dataset"]
        if ds["test_path"] is None and ds["holdout_test"] is None:
            _fail("dataset", "--final-eval necesita dataset.test_path o dataset.holdout_test")
        _check_str(config["selected_from"], "selected_from")
        if config["sweep"]:
            _fail("sweep", "la config de --final-eval es una sola: sin sweep")
        final = _section(config["final"], "final", _FINAL)
        _check_choice(final["mode"], "final.mode", FINAL_MODES)
        _check_int(final["epochs"], "final.epochs", minimum=1, optional=True)
        _check_number(final["threshold"], "final.threshold", optional=True)
        config["final"] = final
    else:
        for key in ("selected_from", "final"):
            if config[key] is not None:
                _fail(key, "solo se acepta con --final-eval")

    _check_consistency(config)
    if run and config["fold"] is not None:
        split = config["dataset"]["split"]
        if split["kind"] != "kfold":
            _fail("fold", "solo existe con dataset.split.kind 'kfold'")
        _check_int(config["fold"], "fold", minimum=0)
        if config["fold"] >= split["k"]:
            _fail("fold", f"tiene que estar en [0, {split['k']}), llegó {config['fold']}")
    return config


def resolve_config(raw: Mapping[str, Any], *, final_eval: bool = False) -> dict[str, Any]:
    """Config base resuelta (defaults + validación) desde un dict. No modifica raw."""
    return _resolve(raw, final_eval=final_eval, run=False)


def resolve_run_config(raw: Mapping[str, Any], *, final_eval: bool = False) -> dict[str, Any]:
    """Valida una config de corrida (la de expand, con seed y fold) y la devuelve resuelta."""
    return _resolve(raw, final_eval=final_eval, run=True)


def _read_json(path: str | PathLike) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError(f"{path}: JSON inválido ({e})") from None


def load_config(path: str | PathLike, *, final_eval: bool = False) -> dict[str, Any]:
    """Lee un JSON de experimento y lo devuelve resuelto y validado.

    Valida también cada combinación del sweep, así un error aparece antes de
    entrenar la primera corrida.
    """
    config = resolve_config(_read_json(path), final_eval=final_eval)
    expand(config)
    return config


def load_run_config(path: str | PathLike) -> dict[str, Any]:
    """Lee el config.json de una corrida (results/<run_name>/<run_id>/config.json)."""
    raw = _read_json(path)
    return resolve_run_config(
        raw, final_eval=isinstance(raw, Mapping) and raw.get("final") is not None
    )


# --- Barridos y hash ---


def _get(config: Mapping[str, Any], key: str, path: str = "") -> Any:
    node: Any = config
    for part in key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            _fail(path or key, f"la clave {key!r} no existe en la config")
        node = node[part]
    return node


def get_path(config: Mapping[str, Any], key: str) -> Any:
    """Valor de una clave en notación punto, p. ej. get_path(c, "training.optimizer.lr")."""
    return _get(config, key)


def set_path(config: dict[str, Any], key: str, value: Any) -> None:
    """Asigna una clave en notación punto; los objetos intermedios tienen que existir."""
    *parents, last = key.split(".")
    node: Any = config
    for i, part in enumerate(parents):
        if not isinstance(node, dict) or not isinstance(node.get(part), dict):
            _fail(
                ".".join(parents[: i + 1]),
                f"no es un objeto de la config: no se puede asignar {key!r}",
            )
        node = node[part]
    node[last] = value


def expand(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Una config de corrida por combinación del sweep × semilla [× fold si es kfold].

    El sweep es un producto cartesiano de claves en notación punto. Cada
    corrida tiene "seed" y "fold" (None salvo kfold) y no tiene "seeds".
    """
    final_eval = config.get("final") is not None
    keys = list(config["sweep"])
    runs: list[dict[str, Any]] = []
    for values in itertools.product(*(config["sweep"][k] for k in keys)):
        combo = copy.deepcopy(dict(config))
        try:
            for key, value in zip(keys, values, strict=True):
                set_path(combo, key, copy.deepcopy(value))
            base = resolve_config(combo, final_eval=final_eval)
        except ConfigError as e:
            raise ConfigError(f"sweep {dict(zip(keys, values, strict=True))}: {e}") from None
        split = base["dataset"]["split"]
        folds = list(range(split["k"])) if split["kind"] == "kfold" else [None]
        for seed in base["seeds"]:
            for fold in folds:
                run = {"run_name": base["run_name"], "seed": seed, "fold": fold}
                run.update({k: v for k, v in base.items() if k not in ("run_name", "seeds")})
                runs.append(copy.deepcopy(run))
    ids = [run_id(r) for r in runs]
    if len(set(ids)) != len(ids):
        _fail("sweep", "dos combinaciones dan la misma config: revisar valores equivalentes")
    return runs


def config_hash(config: Mapping[str, Any]) -> str:
    """sha1 (8 hex) de la config resuelta sin semilla, fold, nombre, sweep ni logging."""
    payload = {k: v for k, v in config.items() if k not in HASH_EXCLUDED}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def run_id(config: Mapping[str, Any]) -> str:
    """Nombre de la carpeta de una corrida: <hash8>_s<seed>[_f<fold>]."""
    fold = config.get("fold")
    return f"{config_hash(config)}_s{config['seed']}" + ("" if fold is None else f"_f{fold}")


# --- Construcción ---


def make_activation(name: str, beta: float) -> Activation:
    """Activación por nombre; beta se pasa solo a las que lo aceptan (tanh, sigmoid)."""
    if "beta" in _signature(ACTIVATIONS[name]):
        return get_activation(name, beta=beta)
    return get_activation(name)


def build(
    config: Mapping[str, Any],
    rng: np.random.Generator,
    *,
    n_features: int | None = None,
    n_outputs: int | None = None,
    task: str | None = None,
    threshold: float | None = None,
) -> tuple[
    Network,
    Optimizer | None,
    Loss,
    list[Callback],
    dict[str, Metric],
    Augmentation | None,
]:
    """(network, optimizer, loss, callbacks, metrics, augment) desde una config resuelta.

    n_features resuelve layers[0] = "auto" y, con n_outputs, verifica que las
    capas coincidan con los datos ya preparados. task y threshold arman las
    métricas con get_metric (threshold None = 0.5). callbacks son los de la
    config (EarlyStopping, AdaptiveEta), en ese orden; el runner agrega los
    suyos. Con trainer "perceptron" no hay optimizador: fit_perceptron usa lr.
    """
    model, training = config["model"], config["training"]
    layers = list(model["layers"])
    if layers[0] == "auto":
        if n_features is None:
            _fail("model.layers[0]", "'auto' necesita saber la cantidad de features")
        layers[0] = n_features
    elif n_features is not None and layers[0] != n_features:
        _fail("model.layers[0]", f"vale {layers[0]} y los datos tienen {n_features} features")
    if n_outputs is not None and layers[-1] != n_outputs:
        _fail(
            "model.layers[-1]",
            f"vale {layers[-1]} y el target codificado tiene {n_outputs} columnas",
        )

    network = Network(
        layers,
        hidden_activation=make_activation(model["hidden_activation"], model["beta"]),
        output_activation=make_activation(model["output_activation"], model["beta"]),
        initializer=get_initializer(model["initializer"], **model["init_params"]),
        rng=rng,
    )
    loss = get_loss(training["loss"])
    optimizer = None
    if config["trainer"] == "backprop":
        params = dict(training["optimizer"])
        optimizer = get_optimizer(params.pop("kind"), **params)

    callbacks: list[Callback] = []
    if training["early_stopping"] is not None:
        callbacks.append(EarlyStopping(**training["early_stopping"]))
    if training["adaptive_eta"] is not None:
        callbacks.append(AdaptiveEta(optimizer, **training["adaptive_eta"]))

    metrics: dict[str, Metric] = {}
    if config["metrics"]:
        if task is None:
            _fail("metrics", "hace falta saber la tarea (binary, multiclass, regression)")
        th = 0.5 if threshold is None else threshold
        metrics = {name: get_metric(name, task=task, threshold=th) for name in config["metrics"]}

    augmentation = training["augmentation"]
    augment = None if augmentation is None else get_augmentation(augmentation)
    return network, optimizer, loss, callbacks, metrics, augment
