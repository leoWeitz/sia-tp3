"""Runner de experimentos: config JSON → entrenamiento → results/.

    python -m experiments.runner <config.json> [--smoke] [--force] [--only CLAVE=VALOR]
                                               [--workers N] [--results-dir DIR]
    python -m experiments.runner --resume <run_dir>
    python -m experiments.runner <config.json> --final-eval [--force]

Cada corrida de expand(config) escribe results/<run_name>/<run_id>/ (CLAUDE.md
§7): config.json, history.csv, metrics.json, predictions.npz (si
logging.save_predictions), model.npz (si logging.save_model) y
weights_history.npz, los pesos de cada época (si
logging.save_weights_history; para redes chicas). Mientras
entrena guarda checkpoint.npz cada logging.checkpoint_every épocas y lo borra
al terminar. metrics.json se escribe último: una corrida que lo tiene está
completa y un barrido relanzado la saltea (salvo --force).
results/<run_name>/log.txt registra cada corrida (i/N) y el tiempo restante.

Datos (docs/03-arquitectura.md §4): se carga el dataset, se excluyen los
índices de dataset.holdout_test (el test sagrado, Constitución C5), el resto
se particiona con un generador propio (semilla de la corrida o split.seed) y
prepare_fold ajusta la normalización solo con train. El modelo se construye
con np.random.default_rng(seed), que decide los pesos iniciales y el orden de
los lotes.

Reanudar (--resume): se reconstruye todo desde config.json y se pisan los
pesos, el estado del optimizador, el del rng de la red y el de los callbacks
con los del checkpoint. fit vuelve a contar desde la época 1 y desde 0 s, así
que _RunTracker les pasa a los callbacks la época y el tiempo absolutos y une
los historiales. Sin augmentation el resultado es idéntico a no haber
cortado. Con augmentation no lo es bit a bit: fit crea en cada llamada un
generador hijo del de la red (rng.spawn) cuyo estado no se guarda, así que el
ruido de las épocas reanudadas es otro (igual de válido).
"""

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from core.callbacks import Callback, EarlyStopping, PrintProgress
from core.history import History
from core.losses import get_loss
from core.metrics import (
    Metric,
    classification_summary,
    get_metric,
    mae,
    mse,
    rmse,
    to_labels,
)
from core.network import Network
from core.perceptron import error_rate, fit_perceptron
from core.serialization import Checkpoint, load_checkpoint, save_checkpoint
from data.loaders import Dataset, load_csv, load_digits_csv
from data.preprocess import TargetScaler, one_hot, scaler_from_state_dict
from data.splits import (
    holdout,
    kfold,
    prepare_fold,
    stratified_holdout,
    stratified_kfold,
    stratify_labels,
)
from data.synthetic import and_dataset, line_samples, xor_dataset
from experiments.config import (
    ONEHOT_ENCODINGS,
    ConfigError,
    build,
    config_hash,
    expand,
    get_path,
    load_config,
    load_run_config,
    resolve_config,
    run_id,
)

SMOKE_EPOCHS = 5
SMOKE_MAX_SAMPLES = 500
SMOKE_DIR = "_smoke"
CHECKPOINT = "checkpoint.npz"
WEIGHTS_HISTORY = "weights_history.npz"

# Cada generador de datos sale de [semilla, stream]: así la partición, la
# submuestra del smoke y el test no comparten números con el modelo, que usa
# default_rng(seed). Cambiar la arquitectura no cambia la partición.
_SPLIT_STREAM, _SMOKE_STREAM, _TEST_STREAM = 1, 2, 3

# Imagen de la activación de salida, para target_encoding "scale_to_output".
OUTPUT_RANGES = MappingProxyType({"tanh": (-1.0, 1.0), "sigmoid": (0.0, 1.0)})
_LINE_FUNCTIONS = MappingProxyType({"identity": lambda x: x, "tanh": np.tanh})

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CODE_DIRS = ("core", "data", "experiments", "analysis")


class RunnerError(RuntimeError):
    """Algo impide correr lo pedido (falta un checkpoint, el test ya se evaluó, etc.)."""


# --- Utilidades ---


def git_commit() -> str:
    """Commit corto del repo, con "-dirty" si hay cambios sin commitear en el código.

    "no-git" si no hay git o el código no está en un repo.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, check=True, timeout=30
        ).stdout.strip()

    try:
        commit = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no", "--", *_CODE_DIRS)
    except (OSError, subprocess.SubprocessError):
        return "no-git"
    return commit + ("-dirty" if dirty else "")


def _write_json(path: Path, data: Any) -> None:
    """Escribe JSON con indentación, primero a un temporal: nunca queda a medio escribir."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class _Log:
    """Agrega líneas con fecha a un log.txt y las repite por write."""

    def __init__(self, path: Path, write: Callable[[str], None]) -> None:
        self.path = path
        self.write = write

    def __call__(self, message: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} | {message}\n")
        self.write(message)


# --- Datos ---


def _synthetic_dataset(spec: Mapping[str, Any]) -> Dataset:
    name = spec["name"]
    if name == "and":
        data = and_dataset()
    elif name == "xor":
        data = xor_dataset()
    else:
        rng = np.random.default_rng(spec["seed"])
        f = _LINE_FUNCTIONS[spec["function"]]
        data = line_samples(f, spec["n"], spec["low"], spec["high"], rng)
        data.meta["function"] = spec["function"]
    content = np.ascontiguousarray(data.X).tobytes() + np.ascontiguousarray(data.y).tobytes()
    data.meta["sha256"] = hashlib.sha256(content).hexdigest()
    data.meta["n_rows"] = len(data.y)
    return data


def load_dataset(ds: Mapping[str, Any], path: str | None = None) -> Dataset:
    """Dataset crudo de dataset.synthetic, o del CSV (path o dataset.path) según dataset.format."""
    if ds["synthetic"] is not None:
        return _synthetic_dataset(ds["synthetic"])
    path = ds["path"] if path is None else path
    if ds["format"] == "digits":
        return load_digits_csv(path)
    return load_csv(
        path,
        ds["target"],
        features=ds["features"],
        drop=ds["drop"],
        categorical=ds["categorical"],
        na_policy=ds["na_policy"],
    )


def _dataset_key(ds: Mapping[str, Any]) -> str:
    """Lo que define qué se lee del disco: dos corridas con la misma clave comparten datos."""
    keys = ("format", "path", "synthetic", "target", "features", "drop", "categorical")
    return json.dumps({k: ds[k] for k in (*keys, "na_policy")}, sort_keys=True)


def infer_task(ds: Mapping[str, Any], y: np.ndarray) -> str:
    """dataset.task; con "auto": multiclass si es one-hot, binary con 2 valores, si no regression.

    Solo mira el conjunto de etiquetas, no estadísticos: se puede calcular sobre todo el dataset.
    """
    if ds["task"] != "auto":
        return ds["task"]
    if ds["target_encoding"] in ONEHOT_ENCODINGS:
        return "multiclass"
    return "binary" if np.unique(y).size == 2 else "regression"


def _strata(y: np.ndarray, task: str) -> np.ndarray:
    """Etiquetas para estratificar: las clases, o bins por cuantiles si el target es continuo."""
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    return stratify_labels(y) if task == "regression" else y


def holdout_test_indices(dataset: Dataset, ds: Mapping[str, Any], task: str) -> np.ndarray:
    """Índices (ordenados) de las filas de test de dataset.holdout_test.

    Si indices_path existe se leen de ahí (así todas las configs usan el mismo
    test); si no, se generan con holdout_test.seed y, si hay indices_path, se
    guardan. Nunca dependen de la semilla de la corrida.
    """
    spec = ds["holdout_test"]
    n = len(dataset.y)
    path = spec["indices_path"]
    if path is not None and Path(path).exists():
        idx = np.load(path)
        if idx.ndim != 1 or idx.size == 0 or np.unique(idx).size != idx.size:
            raise RunnerError(f"{path}: no es una lista de índices distintos")
        if idx.min() < 0 or idx.max() >= n:
            raise RunnerError(f"{path}: hay índices fuera de [0, {n}) (¿cambió el dataset?)")
        return np.sort(idx)
    rng = np.random.default_rng([spec["seed"], _TEST_STREAM])
    dev_ratio = 1.0 - spec["ratio"]
    if spec["stratified"]:
        _, idx = stratified_holdout(_strata(dataset.y, task), dev_ratio, rng)
    else:
        _, idx = holdout(n, dev_ratio, rng)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            np.save(f, idx)
    return idx


def split_positions(
    y_dev: np.ndarray, split: Mapping[str, Any], seed: int, fold: int | None, task: str
) -> tuple[np.ndarray, np.ndarray]:
    """(posiciones de train, de validación) dentro del conjunto de desarrollo."""
    n = len(y_dev)
    if split["kind"] == "none":
        return np.arange(n), np.array([], dtype=np.int64)
    split_seed = seed if split["seed"] is None else split["seed"]
    rng = np.random.default_rng([split_seed, _SPLIT_STREAM])
    if split["kind"] == "holdout":
        if split["stratified"]:
            return stratified_holdout(_strata(y_dev, task), split["ratio"], rng)
        return holdout(n, split["ratio"], rng)
    if split["stratified"]:
        folds = stratified_kfold(_strata(y_dev, task), split["k"], rng)
    else:
        folds = kfold(n, split["k"], rng)
    return folds[fold]


def binary_threshold(y_encoded: np.ndarray, output_activation: str) -> float:
    """Umbral de las métricas binarias: punto medio de los dos valores del target codificado.

    ±1 → 0 (salida tanh), 0/1 → 0.5 (sigmoide). Si el target no tiene
    exactamente dos valores (p. ej. probabilidades), el medio de la imagen de
    la salida: 0 para tanh, 0.5 para el resto.
    """
    values = np.unique(y_encoded)
    if values.size == 2:
        return float(values.mean())
    return 0.0 if output_activation == "tanh" else 0.5


@dataclass
class RunData:
    """Datos de una corrida listos para fit. X_* (n, n_features), y_* (n, n_salidas).

    X_val e y_val son None sin validación (split "none"). idx_* son las filas
    del dataset original. info va a metrics.json.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray | None
    y_val: np.ndarray | None
    idx_train: np.ndarray
    idx_val: np.ndarray
    fitted: dict[str, Any]
    task: str
    threshold: float | None
    n_classes: int | None
    info: dict[str, Any]


def prepare_run_data(
    config: Mapping[str, Any], *, smoke: bool = False, dataset: Dataset | None = None
) -> RunData:
    """Carga (o reusa dataset), excluye el test, particiona y normaliza con train."""
    ds = config["dataset"]
    data = load_dataset(ds) if dataset is None else dataset
    n = len(data.y)
    y = np.asarray(data.y)
    task = infer_task(ds, y)

    test_idx = np.array([], dtype=np.int64)
    if ds["holdout_test"] is not None:
        test_idx = holdout_test_indices(data, ds, task)
    dev_idx = np.setdiff1d(np.arange(n), test_idx)
    n_dev = len(dev_idx)
    if smoke and n_dev > SMOKE_MAX_SAMPLES:
        rng = np.random.default_rng([config["seed"], _SMOKE_STREAM])
        dev_idx = np.sort(rng.choice(dev_idx, SMOKE_MAX_SAMPLES, replace=False))

    pos_train, pos_val = split_positions(
        y[dev_idx], ds["split"], config["seed"], config["fold"], task
    )
    idx_train, idx_val = dev_idx[pos_train], dev_idx[pos_val]
    output = config["model"]["output_activation"]
    X_train, y_train, X_val, y_val, fitted = prepare_fold(
        data,
        idx_train,
        idx_val,
        normalize=ds["normalize"],
        target_encoding=ds["target_encoding"],
        n_classes=ds["n_classes"],
        out_range=OUTPUT_RANGES.get(output),
        scaler_params=None
        if ds["feature_range"] is None
        else {"feature_range": ds["feature_range"]},
        target_in_range=None if ds["target_in_range"] is None else tuple(ds["target_in_range"]),
    )

    has_val = len(idx_val) > 0
    y_all = np.vstack([y_train, y_val]) if has_val else y_train
    threshold = binary_threshold(y_all, output) if task == "binary" else None
    n_classes = {"multiclass": y_train.shape[1], "binary": 2}.get(task)
    info = {
        "path": data.meta.get("path"),
        "sha256": data.meta.get("sha256"),
        "n_rows": n,
        "n_dev": n_dev,
        "n_test_excluded": len(test_idx),
        "n_train": len(idx_train),
        "n_val": len(idx_val),
    }
    return RunData(
        X_train,
        y_train,
        X_val if has_val else None,
        y_val if has_val else None,
        idx_train,
        idx_val,
        fitted,
        task,
        threshold,
        n_classes,
        info,
    )


def encode_targets(y: np.ndarray, preprocessing: Mapping[str, Any]) -> np.ndarray:
    """Codifica un target crudo igual que en el entrenamiento (para evaluar en test)."""
    encoding = preprocessing["target_encoding"]
    if encoding in ONEHOT_ENCODINGS:
        neg = -1.0 if encoding == "pm1_onehot" else 0.0
        return one_hot(y, preprocessing["n_classes"], neg=neg)
    y = np.asarray(y, dtype=np.float64).reshape(len(y), -1)
    if encoding == "scale_to_output":
        return TargetScaler.from_state_dict(preprocessing["target_scaler"]).transform(y)
    return y


def _preprocessing_state(config: Mapping[str, Any], data: RunData) -> dict[str, Any]:
    """Todo lo necesario para aplicar el modelo guardado a datos crudos nuevos."""
    target_scaler = data.fitted["target_scaler"]
    return {
        "scaler": data.fitted["scaler"].state_dict(),
        "target_encoding": config["dataset"]["target_encoding"],
        "n_classes": data.fitted["n_classes"],
        "target_scaler": None if target_scaler is None else target_scaler.state_dict(),
        "task": data.task,
        "threshold": data.threshold,
        "trainer": config["trainer"],
        "loss": config["training"]["loss"],
    }


# --- Evaluación ---


def evaluate_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    task: str,
    threshold: float | None,
    n_classes: int | None,
    loss: Callable[[np.ndarray, np.ndarray], float] | None = None,
    metrics: Mapping[str, Metric] | None = None,
) -> dict[str, Any]:
    """Métricas finales de un conjunto, serializables a JSON.

    y_true (targets codificados) e y_pred (salidas de la red): (n, n_salidas).
    loss y cada métrica de metrics sobre las salidas crudas; además
    classification_summary (binary/multiclass) o mse, rmse y mae (regression).
    """
    out: dict[str, Any] = {}
    if loss is not None:
        out["loss"] = float(loss(y_true, y_pred))
    for name, fn in (metrics or {}).items():
        out[name] = float(fn(y_true, y_pred))
    if task == "regression":
        out.update(mse=mse(y_true, y_pred), rmse=rmse(y_true, y_pred), mae=mae(y_true, y_pred))
    else:
        th = 0.5 if threshold is None else threshold
        labels_true = to_labels(y_true, task=task, threshold=th)
        labels_pred = to_labels(y_pred, task=task, threshold=th)
        out.update(classification_summary(labels_true, labels_pred, n_classes))
    return out


def _prediction_arrays(
    y_true: np.ndarray, y_pred: np.ndarray, task: str, threshold: float | None
) -> dict[str, np.ndarray]:
    arrays = {"y_true": y_true, "y_score": y_pred}
    if task != "regression":
        th = 0.5 if threshold is None else threshold
        arrays["labels_true"] = to_labels(y_true, task=task, threshold=th)
        arrays["labels_pred"] = to_labels(y_pred, task=task, threshold=th)
    return arrays


# --- Una corrida ---


class _RunTracker(Callback):
    """Envuelve los callbacks de una corrida (fit recibe solo este).

    Les pasa la época y el tiempo absolutos de la corrida (al reanudar, fit
    vuelve a contar desde la época 1 y desde 0 s), guarda el historial
    completo en records y, después de on_train_begin (que reinicia a cada
    callback), restaura el estado guardado de los que lo tengan.
    """

    def __init__(
        self,
        callbacks: Iterable[Callback],
        *,
        epoch_offset: int = 0,
        time_offset: float = 0.0,
        records: Iterable[Mapping[str, Any]] = (),
        states: Iterable[tuple[Callback, Mapping[str, Any]]] = (),
    ) -> None:
        self.callbacks = list(callbacks)
        self.epoch_offset = epoch_offset
        self.time_offset = time_offset
        self.records = [dict(r) for r in records]
        self.states = list(states)

    def on_train_begin(self, network: Network) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(network)
        for cb, state in self.states:
            cb.load_state_dict(state)

    def on_epoch_end(self, epoch: int, logs: dict[str, float], network: Network) -> bool:
        epoch += self.epoch_offset
        logs = {**logs, "epoch": epoch, "elapsed_s": logs["elapsed_s"] + self.time_offset}
        self.records.append(dict(logs))
        stop = [cb.on_epoch_end(epoch, logs, network) for cb in self.callbacks]
        return any(stop)

    def on_train_end(self, network: Network) -> None:
        for cb in self.callbacks:
            cb.on_train_end(network)


class _WeightsHistory(Callback):
    """Pesos de la red al final de cada época (logging.save_weights_history).

    Va dentro de _RunTracker, así que recibe épocas absolutas. El primer
    registro son los pesos con los que arranca el entrenamiento (época
    start_epoch: 0, o la del checkpoint al reanudar sin historial guardado).
    state es lo que devolvió arrays() en un checkpoint: al reanudar, el
    historial sigue desde ahí. Guarda una copia de todos los pesos por época:
    pensado para redes chicas (AND, XOR, rectas).
    """

    def __init__(self, start_epoch: int = 0, state: Mapping[str, Any] | None = None) -> None:
        self.start_epoch = start_epoch
        self.epochs: list[int] = []
        self.weights: list[list[tuple[np.ndarray, np.ndarray]]] = []
        if state is not None:
            n_layers = sum(key.startswith("W_") for key in state)
            for i, epoch in enumerate(state["epoch"]):
                self.epochs.append(int(epoch))
                self.weights.append(
                    [
                        (np.array(state[f"W_{j}"][i]), np.array(state[f"b_{j}"][i]))
                        for j in range(n_layers)
                    ]
                )

    def _record(self, epoch: int, network: Network) -> None:
        self.epochs.append(epoch)
        self.weights.append([(layer.W.copy(), layer.b.copy()) for layer in network.layers])

    def on_train_begin(self, network: Network) -> None:
        if not self.epochs:
            self._record(self.start_epoch, network)

    def on_epoch_end(self, epoch: int, logs: dict[str, float], network: Network) -> bool:
        self._record(epoch, network)
        return False

    def arrays(self) -> dict[str, np.ndarray]:
        """{"epoch": (n,), "W_<i>": (n, n_in, n_out), "b_<i>": (n, 1, n_out)}; n = registros."""
        out = {"epoch": np.array(self.epochs, dtype=np.int64)}
        for j in range(len(self.weights[0])):
            out[f"W_{j}"] = np.stack([layers[j][0] for layers in self.weights])
            out[f"b_{j}"] = np.stack([layers[j][1] for layers in self.weights])
        return out


def train_run(
    config: Mapping[str, Any],
    run_dir: str | PathLike,
    *,
    resume: bool = False,
    smoke: bool = False,
    write: Callable[[str], None] = print,
    dataset: Dataset | None = None,
    commit: str | None = None,
    callbacks: Sequence[Callback] = (),
) -> dict[str, Any]:
    """Entrena una corrida (una config de expand) y escribe todos sus archivos en run_dir.

    resume=True sigue desde run_dir/checkpoint.npz; si no, run_dir se vacía y
    se arranca de cero. smoke limita épocas y muestras. dataset evita releer
    el CSV en un barrido; callbacks se agregan al final (p. ej. en tests).
    Devuelve {"run_id", "run_dir", "status", "seconds"}.
    """
    start = time.perf_counter()
    run_dir = Path(run_dir)
    # Identidad de la corrida: la de la config pedida, aunque el smoke recorte épocas.
    rid, run_hash = run_id(config), config_hash(config)
    if smoke:
        config = copy.deepcopy(dict(config))
        config["training"]["epochs"] = min(config["training"]["epochs"], SMOKE_EPOCHS)
    training, logging = config["training"], config["logging"]
    perceptron = config["trainer"] == "perceptron"

    data = prepare_run_data(config, smoke=smoke, dataset=dataset)
    network, optimizer, loss, config_callbacks, metrics, augment = build_for(config, data)

    checkpoint_path = run_dir / CHECKPOINT
    epoch_offset, time_offset, records, states = 0, 0.0, [], []
    weights_state = None
    if resume:
        saved_net, saved_opt, extra = load_checkpoint(checkpoint_path)
        for p, saved in zip(network.params, saved_net.params, strict=True):
            p[...] = saved
        network.rng.bit_generator.state = saved_net.rng.bit_generator.state
        if optimizer is not None:
            optimizer.load_state_dict(saved_opt.state_dict())
        epoch_offset, time_offset = int(extra["epoch"]), float(extra["elapsed_s"])
        records = extra["history"]
        states = list(zip(config_callbacks, extra["callbacks"], strict=True))
        weights_state = extra.get("weights_history")
    else:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        _write_json(run_dir / "config.json", config)

    tracker = _RunTracker(
        [*config_callbacks, PrintProgress(logging["every"], write)],
        epoch_offset=epoch_offset,
        time_offset=time_offset,
        records=records,
        states=states,
    )
    weights = None
    if logging["save_weights_history"]:
        # Antes de Checkpoint: el checkpoint de una época ya incluye sus pesos.
        weights = _WeightsHistory(epoch_offset, weights_state)
        tracker.callbacks.append(weights)
    if logging["checkpoint_every"]:

        def checkpoint_extra(epoch: int, logs: dict[str, float]) -> dict[str, Any]:
            return {
                "epoch": epoch,
                "elapsed_s": logs["elapsed_s"],
                "history": tracker.records,
                "callbacks": [cb.state_dict() for cb in config_callbacks],
                "weights_history": None if weights is None else weights.arrays(),
                "smoke": smoke,
            }

        tracker.callbacks.append(
            Checkpoint(checkpoint_path, logging["checkpoint_every"], optimizer, checkpoint_extra)
        )
    tracker.callbacks.extend(callbacks)

    epochs = training["epochs"]
    remaining = epochs - epoch_offset
    diverged_before = bool(records) and not np.isfinite(records[-1]["train_loss"])
    stopped_before = any(state.get("stopped_epoch") is not None for _, state in states)
    validation = None if data.X_val is None else (data.X_val, data.y_val)
    if remaining <= 0 or diverged_before or stopped_before:
        # El checkpoint es del final del entrenamiento: solo falta on_train_end
        # (p. ej. que EarlyStopping restaure la mejor época).
        tracker.on_train_begin(network)
        tracker.on_train_end(network)
    elif perceptron:
        fit_perceptron(
            network,
            data.X_train,
            data.y_train,
            lr=training["optimizer"]["lr"],
            epochs=remaining,
            batch_size=training["batch_size"],
            callbacks=[tracker],
        )
    else:
        network.fit(
            data.X_train,
            data.y_train,
            optimizer,
            loss,
            remaining,
            batch_size=training["batch_size"],
            validation=validation,
            callbacks=[tracker],
            metrics=metrics,
            l2=training["l2"],
            augment=augment,
        )

    records = tracker.records
    history = History()
    for record in records:
        history.append(record)
    history.to_csv(run_dir / "history.csv")
    status = "diverged" if records and not np.isfinite(records[-1]["train_loss"]) else "ok"

    loss_fn = error_rate if perceptron else loss.value
    scores = {"task": data.task, "threshold": data.threshold, "n_classes": data.n_classes}
    y_pred_train = network.predict(data.X_train)
    train_scores = evaluate_outputs(
        data.y_train, y_pred_train, loss=loss_fn, metrics=metrics, **scores
    )
    val_scores, y_pred_val = None, None
    if validation is not None:
        y_pred_val = network.predict(data.X_val)
        val_scores = evaluate_outputs(
            data.y_val, y_pred_val, loss=loss_fn, metrics=metrics, **scores
        )

    early = next((cb for cb in config_callbacks if isinstance(cb, EarlyStopping)), None)
    if early is not None:
        best_epoch = early.best_epoch
    elif validation is not None and records:
        val_loss = np.array([r["val_loss"] for r in records], dtype=float)
        best_epoch = (
            int(records[int(np.nanargmin(val_loss))]["epoch"])
            if np.isfinite(val_loss).any()
            else None
        )
    else:
        best_epoch = int(records[-1]["epoch"]) if records else None
    time_total = float(records[-1]["elapsed_s"]) if records else 0.0

    if logging["save_predictions"]:
        if validation is not None:
            subset, y_true, y_pred, idx = "val", data.y_val, y_pred_val, data.idx_val
        else:
            subset, y_true, y_pred, idx = "train", data.y_train, y_pred_train, data.idx_train
        arrays = _prediction_arrays(y_true, y_pred, data.task, data.threshold)
        np.savez(run_dir / "predictions.npz", idx=idx, subset=np.array(subset), **arrays)
    if logging["save_model"]:
        extra = {
            "config": config,
            "epoch": records[-1]["epoch"] if records else 0,
            "preprocessing": _preprocessing_state(config, data),
        }
        save_checkpoint(run_dir / "model.npz", network, optimizer, extra)
    if weights is not None:
        np.savez(run_dir / WEIGHTS_HISTORY, **weights.arrays())

    metrics_json = {
        "run_id": rid,
        "hash": run_hash,
        "seed": config["seed"],
        "fold": config["fold"],
        "status": status,
        "trainer": config["trainer"],
        **scores,
        "epochs": epochs,
        "epochs_trained": len(records),
        "best_epoch": best_epoch,
        "stopped_epoch": None if early is None else early.stopped_epoch,
        "time_total_s": time_total,
        "s_per_epoch": time_total / len(records) if records else None,
        "n_params": network.n_params,
        "lr_final": float(optimizer.lr if optimizer is not None else training["optimizer"]["lr"]),
        "resumed_from_epoch": epoch_offset if resume else None,
        "smoke": smoke,
        "git_commit": git_commit() if commit is None else commit,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "data": data.info,
        "train": train_scores,
        "val": val_scores,
    }
    _write_json(run_dir / "metrics.json", metrics_json)
    checkpoint_path.unlink(missing_ok=True)
    return {
        "run_id": rid,
        "run_dir": str(run_dir),
        "status": status,
        "seconds": time.perf_counter() - start,
    }


def build_for(config: Mapping[str, Any], data: RunData) -> tuple:
    """build de experiments.config con las formas, la tarea y el umbral de los datos."""
    return build(
        config,
        np.random.default_rng(config["seed"]),
        n_features=data.X_train.shape[1],
        n_outputs=data.y_train.shape[1],
        task=data.task,
        threshold=data.threshold,
    )


def _safe_run(
    config: Mapping[str, Any],
    run_dir: Path,
    *,
    smoke: bool,
    write: Callable[[str], None],
    commit: str | None,
    dataset: Dataset | None = None,
) -> dict[str, Any]:
    """train_run que, ante un error, lo deja en run_dir/error.txt y devuelve status "error"."""
    start = time.perf_counter()
    try:
        return train_run(config, run_dir, smoke=smoke, write=write, dataset=dataset, commit=commit)
    except Exception:
        error = traceback.format_exc()
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "error.txt").write_text(error, encoding="utf-8")
        return {
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "status": "error",
            "seconds": time.perf_counter() - start,
            "error": error,
        }


def _run_task(config: dict[str, Any], run_dir: str, smoke: bool, commit: str) -> dict[str, Any]:
    """Una corrida en un proceso del pool (--workers). A nivel de módulo por spawn en Windows."""
    rid = Path(run_dir).name

    def write(line: str) -> None:
        print(f"[{rid}] {line}", flush=True)

    return _safe_run(config, Path(run_dir), smoke=smoke, write=write, commit=commit)


# --- Barridos ---


def _parse_only(items: Iterable[str]) -> list[tuple[str, Any]]:
    conditions = []
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ConfigError(f"--only {item!r}: se espera clave=valor (p. ej. seed=0)")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        conditions.append((key.strip(), value))
    return conditions


def filter_runs(runs: Sequence[dict[str, Any]], only: Iterable[str]) -> list[dict[str, Any]]:
    """Las corridas cuyas claves (notación punto) valen lo pedido en cada "clave=valor"."""
    conditions = _parse_only(only)
    selected = [r for r in runs if all(get_path(r, k) == v for k, v in conditions)]
    if conditions and not selected:
        raise ConfigError(f"--only {[f'{k}={v}' for k, v in conditions]}: ninguna corrida coincide")
    return selected


def _warm_up(runs: Sequence[Mapping[str, Any]]) -> None:
    """Antes de repartir corridas entre procesos: crea la caché de dígitos y los índices de test.

    Así los workers no escriben el mismo archivo a la vez.
    """
    seen = set()
    for run in runs:
        ds = run["dataset"]
        key = _dataset_key(ds) + json.dumps(ds["holdout_test"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        data = load_dataset(ds)
        if ds["holdout_test"] is not None:
            holdout_test_indices(data, ds, infer_task(ds, data.y))


def _smoke_estimate(
    result: Mapping[str, Any], runs: Sequence[Mapping[str, Any]], workers: int
) -> str:
    """Estimación del barrido completo desde el smoke: s/época × épocas × corridas."""
    metrics = _read_json(Path(result["run_dir"]) / "metrics.json")
    data = metrics["data"]
    n_smoke = max(data["n_train"], 1)
    split = runs[0]["dataset"]["split"]
    fraction = {"none": 1.0, "holdout": split.get("ratio"), "kfold": None}[split["kind"]]
    if fraction is None:
        fraction = (split["k"] - 1) / split["k"]
    n_full = max(round(data["n_dev"] * fraction), 1)
    s_epoch = metrics["s_per_epoch"] or 0.0
    s_epoch_full = s_epoch * n_full / n_smoke
    total_epochs = sum(r["training"]["epochs"] for r in runs)
    estimate = s_epoch_full * total_epochs / max(workers, 1)
    return (
        f"smoke: {s_epoch:.3g} s/época con {n_smoke} muestras de train, ~{s_epoch_full:.3g} "
        f"s/época con {n_full}; barrido completo: {len(runs)} corridas, {total_epochs} épocas "
        f"en total ≈ {_format_seconds(estimate)} con {workers} worker(s) "
        "(cota superior: no descuenta early stopping)"
    )


def run_experiment(
    config_path: str | PathLike,
    *,
    results_dir: str | PathLike = "results",
    smoke: bool = False,
    force: bool = False,
    only: Iterable[str] = (),
    workers: int = 1,
    write: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Corre todas las corridas de un JSON (o las de --only) y devuelve un resultado por corrida.

    Saltea las que ya tienen metrics.json salvo force. Con smoke corre solo la
    primera, con pocas épocas y muestras, en results/_smoke/, y agrega una
    estimación del tiempo del barrido completo. Un error en una corrida queda
    en su error.txt y no frena el barrido (status "error").
    """
    config = load_config(config_path)
    all_runs = filter_runs(expand(config), only)
    runs = all_runs[:1] if smoke else all_runs
    root = Path(results_dir)
    root = root / SMOKE_DIR / config["run_name"] if smoke else root / config["run_name"]
    force = force or smoke
    commit = git_commit()
    log = _Log(root / "log.txt", write)
    total = len(runs)
    log(
        f"inicio: {total} corrida(s) de {config_path}, workers={workers}, commit={commit}"
        + (", smoke" if smoke else "")
    )

    results: list[dict[str, Any] | None] = [None] * total
    done, durations = 0, []
    todo = []
    for i, run in enumerate(runs):
        run_dir = root / run_id(run)
        if (run_dir / "metrics.json").exists() and not force:
            done += 1
            results[i] = {"run_id": run_dir.name, "run_dir": str(run_dir), "status": "skipped"}
            log(f"[{done}/{total}] {run_dir.name}: salteada (ya tiene metrics.json)")
        else:
            todo.append(i)

    def finished(i: int, result: dict[str, Any]) -> None:
        nonlocal done
        done += 1
        results[i] = result
        durations.append(result["seconds"])
        pending = total - done
        eta = np.mean(durations) * pending / min(max(workers, 1), max(pending, 1))
        detail = f" ({result['error'].strip().splitlines()[-1]})" if "error" in result else ""
        log(
            f"[{done}/{total}] {result['run_id']}: {result['status']}{detail} en "
            f"{result['seconds']:.1f} s | restante estimado {_format_seconds(eta)}"
        )

    if workers > 1 and len(todo) > 1:
        _warm_up([runs[i] for i in todo])
        with ProcessPoolExecutor(max_workers=min(workers, len(todo))) as pool:
            futures = {
                pool.submit(_run_task, runs[i], str(root / run_id(runs[i])), smoke, commit): i
                for i in todo
            }
            for future in as_completed(futures):
                finished(futures[future], future.result())
    else:
        datasets: dict[str, Dataset] = {}
        for i in todo:
            run = runs[i]
            key = _dataset_key(run["dataset"])
            write(f"[{done + 1}/{total}] {run_id(run)}")
            try:
                if key not in datasets:
                    datasets[key] = load_dataset(run["dataset"])
                dataset = datasets[key]
            except Exception:
                dataset = None  # _safe_run lo vuelve a intentar y registra el error
            result = _safe_run(
                run,
                root / run_id(run),
                smoke=smoke,
                write=lambda line: write("  " + line),
                commit=commit,
                dataset=dataset,
            )
            finished(i, result)

    counts = {s: sum(r["status"] == s for r in results) for s in ("ok", "diverged", "error")}
    counts["salteadas"] = sum(r["status"] == "skipped" for r in results)
    log("fin: " + ", ".join(f"{n} {s}" for s, n in counts.items()))
    if smoke and results and results[0]["status"] != "error":
        write(_smoke_estimate(results[0], all_runs, workers))
    return results


def resume_run(run_dir: str | PathLike, *, write: Callable[[str], None] = print) -> dict[str, Any]:
    """Sigue una corrida cortada desde su checkpoint.npz hasta las épocas de su config.json."""
    run_dir = Path(run_dir)
    if (run_dir / "metrics.json").exists():
        write(f"{run_dir.name}: ya está completa (tiene metrics.json), no hay nada que reanudar")
        return {"run_id": run_dir.name, "run_dir": str(run_dir), "status": "skipped"}
    if not (run_dir / CHECKPOINT).exists():
        raise RunnerError(
            f"{run_dir}: no hay {CHECKPOINT} para reanudar. Relanzar el barrido sin --resume "
            "rehace la corrida desde cero"
        )
    config = load_run_config(run_dir / "config.json")
    if config["final"] is not None:
        raise RunnerError(f"{run_dir}: las corridas de --final-eval no se reanudan")
    _, _, extra = load_checkpoint(run_dir / CHECKPOINT)
    log = _Log(run_dir.parent / "log.txt", write)
    log(f"reanudando {run_dir.name} desde la época {extra['epoch']}")
    result = train_run(config, run_dir, resume=True, smoke=bool(extra.get("smoke")), write=write)
    log(f"{run_dir.name}: {result['status']} en {result['seconds']:.1f} s (reanudada)")
    return result


# --- Evaluación final ---


def selected_runs(selected_from: str) -> list[Path]:
    """Carpetas elegidas: una corrida, o todas las de un prefijo <run_name>/<hash8>."""
    path = Path(selected_from)
    if (path / "metrics.json").exists():
        return [path]
    runs = sorted(p for p in path.parent.glob(path.name + "_*") if (p / "metrics.json").exists())
    if not runs:
        raise RunnerError(
            f"selected_from {selected_from!r}: no hay corridas terminadas. Se espera "
            "results/<run_name>/<run_id> o el prefijo results/<run_name>/<hash8>"
        )
    return runs


def evaluate_model(
    model_path: str | PathLike,
    X: np.ndarray,
    y: np.ndarray,
    *,
    threshold: float | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Aplica un model.npz a datos crudos X (n, n_features), y (n,) → (métricas, predicciones).

    Usa la normalización y la codificación guardadas con el modelo (ajustadas
    con su train). threshold reemplaza al umbral binario del modelo.
    """
    network, _, extra = load_checkpoint(model_path)
    prep = extra["preprocessing"]
    X = scaler_from_state_dict(prep["scaler"]).transform(X)
    y = encode_targets(y, prep)
    y_pred = network.predict(X)
    task = prep["task"]
    th = prep["threshold"] if threshold is None else threshold
    if prep["trainer"] == "perceptron":
        loss_fn = error_rate
    else:
        loss_fn = get_loss(prep["loss"]).value
    metrics = {name: get_metric(name, task=task, threshold=0.5 if th is None else th)
               for name in extra["config"]["metrics"]}  # fmt: skip
    n_classes = y.shape[1] if task == "multiclass" else 2
    scores = evaluate_outputs(
        y, y_pred, task=task, threshold=th, n_classes=n_classes, loss=loss_fn, metrics=metrics
    )
    return scores, _prediction_arrays(y, y_pred, task, th)


def _load_test(config: Mapping[str, Any], dataset: Dataset) -> tuple[np.ndarray, np.ndarray, dict]:
    ds = config["dataset"]
    if ds["test_path"] is not None:
        test = load_dataset(ds, path=ds["test_path"])
        if test.feature_names != dataset.feature_names:
            raise RunnerError(
                f"{ds['test_path']}: las features no coinciden con las de {ds['path']}"
            )
        info = {"source": "test_path", "path": ds["test_path"], "sha256": test.meta["sha256"]}
        return test.X, np.asarray(test.y), {**info, "n": len(test.y)}
    idx = holdout_test_indices(dataset, ds, infer_task(ds, dataset.y))
    info = {
        "source": "holdout_test",
        "path": ds["path"],
        "sha256": dataset.meta.get("sha256"),
        "indices_path": ds["holdout_test"]["indices_path"],
    }
    return dataset.X[idx], np.asarray(dataset.y)[idx], {**info, "n": len(idx)}


def _scalar_summary(evaluations: Sequence[Mapping[str, Any]]) -> tuple[dict, dict]:
    """Media y desvío (ddof = 1; None con un solo modelo) de cada métrica escalar en test."""
    keys = [
        k for k, v in evaluations[0]["test"].items() if isinstance(v, (int, float)) and k != "n"
    ]
    mean, std = {}, {}
    for key in keys:
        values = np.array([e["test"][key] for e in evaluations], dtype=float)
        mean[key] = float(values.mean())
        std[key] = float(values.std(ddof=1)) if len(values) > 1 else None
    return mean, std


def final_eval(
    config_path: str | PathLike,
    *,
    results_dir: str | PathLike = "results",
    force: bool = False,
    write: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Evalúa UNA vez en test la configuración elegida y escribe results/<run_name>/final_eval.json.

    final.mode "retrain": por cada semilla, entrena con todo el conjunto de
    desarrollo (sin validación ni early stopping) durante final.epochs, o la
    mediana de best_epoch de las corridas de selected_from, en
    results/<run_name>/final_s<seed>/. "reuse": usa el model.npz de cada
    corrida de selected_from. Si final_eval.json ya existe es un error salvo force.
    """
    config = load_config(config_path, final_eval=True)
    root = Path(results_dir) / config["run_name"]
    report_path = root / "final_eval.json"
    if report_path.exists() and not force:
        raise RunnerError(
            f"{report_path} ya existe: el test se evalúa una sola vez (--force para rehacerlo)"
        )
    root.mkdir(parents=True, exist_ok=True)
    log = _Log(root / "log.txt", write)
    commit = git_commit()
    final = config["final"]
    selected = selected_runs(config["selected_from"])
    dataset = load_dataset(config["dataset"])

    models: list[tuple[str, Path, Path]] = []  # (etiqueta, model.npz, carpeta de salida)
    epochs = None
    if final["mode"] == "retrain":
        best = [
            m["best_epoch"]
            for m in (_read_json(p / "metrics.json") for p in selected)
            if m["status"] == "ok" and m["best_epoch"] is not None
        ]
        if final["epochs"] is not None:
            epochs = final["epochs"]
        elif best:
            epochs = max(1, int(np.round(np.median(best))))
        else:
            raise RunnerError("ninguna corrida de selected_from terminó ok con best_epoch")
        base = copy.deepcopy(config)
        base["training"]["epochs"] = epochs
        base["training"]["early_stopping"] = None
        base["dataset"]["split"] = {"kind": "none"}
        base["logging"]["save_model"] = True
        log(f"final-eval: reentreno con todo desarrollo durante {epochs} épocas")
        for run in expand(resolve_config(base, final_eval=True)):
            run_dir = root / f"final_s{run['seed']}"
            write(f"final_s{run['seed']}")
            result = train_run(
                run, run_dir, write=lambda line: write("  " + line), dataset=dataset, commit=commit
            )
            log(f"final_s{run['seed']}: {result['status']} en {result['seconds']:.1f} s")
            models.append((f"s{run['seed']}", run_dir / "model.npz", run_dir))
    else:
        for run_dir in selected:
            if not (run_dir / "model.npz").exists():
                raise RunnerError(f"{run_dir}: no tiene model.npz (logging.save_model)")
            models.append((run_dir.name, run_dir / "model.npz", root / f"final_{run_dir.name}"))

    X_test, y_test, test_info = _load_test(config, dataset)
    log(
        f"final-eval: lectura del test ({test_info['source']}: {test_info['path']}, "
        f"n={len(y_test)})"
    )
    evaluations = []
    for label, model_path, out_dir in models:
        scores, arrays = evaluate_model(model_path, X_test, y_test, threshold=final["threshold"])
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "test_predictions.npz", **arrays)
        evaluations.append({"label": label, "model": str(model_path), "test": scores})
    mean, std = _scalar_summary(evaluations)

    report = {
        "run_name": config["run_name"],
        "mode": final["mode"],
        "selected_from": config["selected_from"],
        "selected_runs": [p.name for p in selected],
        "epochs": epochs,
        "threshold": final["threshold"],
        "test": test_info,
        "n_models": len(evaluations),
        "mean": mean,
        "std": std,
        "models": evaluations,
        "git_commit": commit,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(report_path, report)
    log(f"final-eval: {len(evaluations)} modelo(s) evaluados; escrito {report_path}")
    return report


# --- CLI ---


def main(argv: Sequence[str] | None = None) -> int:
    """CLI. Devuelve 0 si todo terminó (ok o diverged), 1 si falló alguna corrida, 2 si hubo
    un error de uso (config inválida, falta un archivo).
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(
        prog="python -m experiments.runner",
        description="Corre las corridas de una config JSON y guarda todo en results/.",
    )
    parser.add_argument("config", nargs="?", help="JSON del experimento (experiments/configs/...)")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="solo la primera corrida, 5 épocas, ≤ 500 muestras, a results/_smoke/",
    )
    parser.add_argument(
        "--resume", metavar="RUN_DIR", help="sigue una corrida cortada desde su checkpoint.npz"
    )
    parser.add_argument(
        "--final-eval", action="store_true", help="evalúa una vez en test la config elegida"
    )
    parser.add_argument(
        "--force", action="store_true", help="rehace corridas completas (o final_eval.json)"
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="CLAVE=VALOR",
        help="solo las corridas con ese valor (repetible)",
    )
    parser.add_argument("--workers", type=int, default=1, help="corridas en paralelo (procesos)")
    parser.add_argument("--results-dir", default="results", help="carpeta raíz de resultados")
    args = parser.parse_args(argv)

    if args.resume is not None:
        if args.config or args.smoke or args.final_eval or args.only or args.workers != 1:
            parser.error("--resume va solo: la config sale de <run_dir>/config.json")
    elif args.config is None:
        parser.error("falta la config JSON (o --resume <run_dir>)")
    if args.final_eval and (args.smoke or args.only or args.workers != 1):
        parser.error("--final-eval no se combina con --smoke, --only ni --workers")
    if args.workers < 1:
        parser.error("--workers tiene que ser >= 1")

    try:
        if args.resume is not None:
            result = resume_run(args.resume)
            return 1 if result["status"] == "error" else 0
        if args.final_eval:
            final_eval(args.config, results_dir=args.results_dir, force=args.force)
            return 0
        results = run_experiment(
            args.config,
            results_dir=args.results_dir,
            smoke=args.smoke,
            force=args.force,
            only=args.only,
            workers=args.workers,
        )
    except (ConfigError, RunnerError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 1 if any(r["status"] == "error" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
