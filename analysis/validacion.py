"""Figuras de validación del motor (F08-T5 y F08-T6): V-01 a V-04.

    python -m analysis.validacion [--results-dir results] [--out-dir figures/validacion]

Solo lee results/ (CLAUDE.md §2.6, Constitución C6). Antes hay que correr
cada config de experiments/configs/validacion/ con el runner:

    python -m experiments.runner experiments/configs/validacion/<config>.json

Las regiones de decisión y las curvas ajustadas cargan el model.npz de cada
corrida y predicen sobre una grilla: eso no entrena. Las curvas son media ±
desvío (ddof = 1) entre semillas, con el n en el título o la leyenda.

Figuras (FIGURES), en <out-dir> como .png y .pdf:
- V-01_and_decision_boundary: recta de decisión del escalón en AND, época por
  época, de la semilla que más épocas tarda en converger (necesita
  logging.save_weights_history).
- V-01_and_error: tasa de error de AND vs época, desde la época 0 (pesos iniciales).
- V-02_linear_fit, V-03_tanh_fit: datos + curva ajustada, y MSE vs época.
- V-04_xor_error_step_vs_mlp: error de XOR vs época, escalón vs MLP en la
  misma unidad: en el escalón train_loss ya es la tasa de error; en el MLP se
  usa 1 − train_accuracy.
- V-04_xor_decision_regions: regiones de decisión del escalón y de los MLP.
- V-04_xor_init_solved (T6): % de semillas que resuelven XOR por
  inicialización y arquitectura. "Resuelve" = accuracy 100 % sobre los 4
  puntos al final del entrenamiento.
- V-04_xor_init_loss (T6): loss vs época por inicialización; con U(±0.1) la
  red queda en la meseta cercana al origen (salida ≈ 0, MSE ≈ 1).

Las corridas Xavier de T6 son las de val_xor_221 y val_xor_2321 (misma
config que xor_init_uniform salvo el inicializador); init_runs verifica que
sean comparables. Además escribe <out-dir>/resumen.json con los números de la
slide de validación y los imprime.
"""

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.common import PALETTE, apply_style, load_histories, load_runs, save_figure
from core.activations import Step
from core.initializers import get_initializer
from core.network import Network
from core.perceptron import error_rate
from core.serialization import load_checkpoint
from data.preprocess import scaler_from_state_dict
from experiments.config import load_run_config
from experiments.runner import load_dataset

CONFIG_DIR = Path(__file__).resolve().parents[1] / "experiments" / "configs" / "validacion"
# Configs de experiments/configs/validacion/ que usa este análisis (nombre sin .json).
VALIDATION_CONFIGS = (
    "and_step",
    "linear_identity",
    "nonlinear_tanh",
    "xor_step",
    "xor_221",
    "xor_2321",
    "xor_init_uniform",
)
FIGURES = (
    "V-01_and_decision_boundary",
    "V-01_and_error",
    "V-02_linear_fit",
    "V-03_tanh_fit",
    "V-04_xor_error_step_vs_mlp",
    "V-04_xor_decision_regions",
    "V-04_xor_init_solved",
    "V-04_xor_init_loss",
)

# Tonos de texto y ejes de analysis/common.py.
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
# Rampa secuencial azul (claro → oscuro) para ordinales como las épocas.
_BLUE_RAMP = (
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)
# Divergente para la salida de la red en XOR: −1 azul, 0 gris neutro, +1 rojo.
_DIVERGING = ("#2a78d6", "#f0efec", "#e34948")
_LIM = 1.6  # los datos lógicos están en [−1, 1]²
_MAX_LINES = 8  # rectas de AND que se dibujan como máximo
_INIT_ORDER = ("U(±0.1)", "U(±0.5)", "U(±1)", "Xavier")
_ARCH_ORDER = ("[2, 2, 1]", "[2, 3, 2, 1]")
# Lo que tiene que coincidir entre las corridas de T6 para compararlas.
_T6_SAME = (
    "trainer",
    "dataset.synthetic.name",
    "model.hidden_activation",
    "model.output_activation",
    "model.beta",
    "training.loss",
    "training.optimizer.kind",
    "training.optimizer.lr",
    "training.batch_size",
    "training.epochs",
)


class MissingResults(FileNotFoundError):
    """Falta correr alguna config de validación con el runner."""


# --- Carga ---


def config_run_name(config: str) -> str:
    """run_name de experiments/configs/validacion/<config>.json."""
    with open(CONFIG_DIR / f"{config}.json", encoding="utf-8") as f:
        return json.load(f)["run_name"]


def check_results(results_dir: str | PathLike) -> None:
    """MissingResults con los comandos a correr si falta alguna de VALIDATION_CONFIGS."""
    missing = [
        c
        for c in VALIDATION_CONFIGS
        if not any((Path(results_dir) / config_run_name(c)).glob("*/metrics.json"))
    ]
    if missing:
        commands = "\n".join(
            f"    python -m experiments.runner experiments/configs/validacion/{c}.json"
            for c in missing
        )
        raise MissingResults(
            f"{results_dir}: faltan corridas de validación. Correr antes:\n{commands}"
        )


def validation_runs(results_dir: str | PathLike, config: str) -> tuple[pd.DataFrame, Path]:
    """(corridas terminadas ordenadas por semilla, carpeta results/<run_name>) de una config."""
    path = Path(results_dir) / config_run_name(config)
    runs = load_runs(path)
    if runs.empty:
        raise MissingResults(
            f"{path}: no hay corridas. Correr antes: "
            f"python -m experiments.runner experiments/configs/validacion/{config}.json"
        )
    return runs.sort_values("seed").reset_index(drop=True), path


def init_label(initializer: str, params: Mapping[str, float]) -> str:
    """Nombre corto de un inicializador: "U(±0.1)", "U(0, 1)", "Xavier"."""
    if initializer == "uniform":
        uniform = get_initializer("uniform", **params)
        if uniform.low == -uniform.high:
            return f"U(±{uniform.high:g})"
        return f"U({uniform.low:g}, {uniform.high:g})"
    return {"xavier": "Xavier", "he": "He"}.get(initializer, initializer)


def init_runs(results_dir: str | PathLike) -> pd.DataFrame:
    """Corridas de T6: uniformes de xor_init_uniform + Xavier de xor_221 y xor_2321.

    Agrega las columnas init (init_label) y path (carpeta de la corrida).
    ValueError si las corridas no son comparables: distinto entrenamiento
    (_T6_SAME) o distintas semillas entre grupos.
    """
    frames = []
    for config in ("xor_init_uniform", "xor_221", "xor_2321"):
        runs, path = validation_runs(results_dir, config)
        runs["path"] = [str(path / run) for run in runs["run"]]
        frames.append(runs)
    runs = pd.concat(frames, ignore_index=True)

    for column in _T6_SAME:
        values = runs.groupby("run_name")[column].first()
        if runs[column].nunique(dropna=False) > 1:
            raise ValueError(
                f"{column}: las corridas de T6 no son comparables ({values.to_dict()}). "
                "Correr xor_221, xor_2321 y xor_init_uniform con el mismo entrenamiento"
            )

    def label(row: pd.Series) -> str:
        params = {
            key: float(row[f"model.init_params.{key}"])
            for key in ("low", "high")
            if f"model.init_params.{key}" in row and pd.notna(row[f"model.init_params.{key}"])
        }
        return init_label(row["model.initializer"], params)

    runs["init"] = runs.apply(label, axis=1)
    seeds = runs.groupby(["model.layers", "init"])["seed"].apply(lambda s: tuple(sorted(s)))
    if seeds.nunique() > 1:
        raise ValueError(
            f"seeds: los grupos de T6 no tienen las mismas semillas ({seeds.to_dict()})"
        )
    return runs


def predict_model(model_path: str | PathLike, X: np.ndarray) -> tuple[np.ndarray, Network]:
    """(salidas (n, n_salidas), red) de un model.npz sobre datos crudos X (n, n_features).

    Aplica la normalización guardada con el modelo. Solo predice: no entrena.
    """
    network, _, extra = load_checkpoint(model_path)
    scaler = scaler_from_state_dict(extra["preprocessing"]["scaler"])
    return network.predict(scaler.transform(X)), network


def run_dataset(run_dir: str | PathLike) -> tuple[np.ndarray, np.ndarray]:
    """(X (n, n_features), y (n, 1)) crudos del dataset de una corrida, desde su config.json."""
    config = load_run_config(Path(run_dir) / "config.json")
    data = load_dataset(config["dataset"])
    return data.X, np.asarray(data.y, dtype=float).reshape(len(data.y), -1)


# --- Cálculos ---


def curve_stats(
    histories: pd.DataFrame,
    metric: str,
    transform: Callable[[pd.Series], pd.Series] | None = None,
) -> pd.DataFrame:
    """Media, desvío (ddof = 1) y n por época de una columna del historial.

    histories: formato largo de load_histories. transform se aplica a cada
    valor antes de agregar (p. ej. 1 − accuracy). Columnas: epoch, mean, std, n.
    """
    rows = histories[histories["metric"] == metric]
    if rows.empty:
        raise ValueError(f"No hay valores de {metric!r} en los historiales")
    values = rows["value"].astype(float)
    if transform is not None:
        values = transform(values)
    by_epoch = rows.assign(value=values).groupby("epoch")["value"]
    stats = pd.DataFrame(
        {"mean": by_epoch.mean(), "std": by_epoch.std(ddof=1), "n": by_epoch.count()}
    )
    return stats.reset_index()


def convergence_epoch(epochs: np.ndarray, errors: np.ndarray) -> int | None:
    """Época desde la que el error es 0 hasta el final; None si termina con error."""
    epochs, errors = np.asarray(epochs), np.asarray(errors, dtype=float)
    if errors.size == 0 or errors[-1] != 0:
        return None
    nonzero = np.flatnonzero(errors != 0)
    return int(epochs[0] if nonzero.size == 0 else epochs[nonzero[-1] + 1])


def run_convergence(histories: pd.DataFrame, metric: str, error: bool = True) -> pd.Series:
    """convergence_epoch de cada corrida (índice run). error=False usa 1 − metric."""
    rows = histories[histories["metric"] == metric].sort_values("epoch")
    out = {}
    for run, group in rows.groupby("run"):
        values = group["value"].to_numpy(dtype=float)
        out[run] = convergence_epoch(group["epoch"].to_numpy(), values if error else 1 - values)
    return pd.Series(out, dtype=object)


def decision_line(w: np.ndarray, b: float, lim: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Dos puntos (xs, ys) de la recta w₁x₁ + w₂x₂ + b = 0 que cruzan el cuadrado [−lim, lim]².

    Se despeja la coordenada con mayor peso, así también sirve para rectas
    verticales u horizontales. None si w = 0 (no hay recta).
    """
    w1, w2 = float(w[0]), float(w[1])
    if w1 == 0 and w2 == 0:
        return None
    if abs(w2) >= abs(w1):
        xs = np.array([-lim, lim])
        return xs, -(w1 * xs + b) / w2
    ys = np.array([-lim, lim])
    return -(w2 * ys + b) / w1, ys


def describe(values: Sequence[float]) -> dict[str, Any]:
    """{n, mean, std (ddof = 1; None con n < 2), min, max} de una lista de números."""
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else None,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _pm(stats: Mapping[str, Any], fmt: str = ".3g") -> str:
    """ "media ± desvío" de describe()."""
    if stats["mean"] is None:
        return "—"
    std = "—" if stats["std"] is None else format(stats["std"], fmt)
    return f"{format(stats['mean'], fmt)} ± {std}"


# --- Gráficos ---


def _band(ax: Any, stats: pd.DataFrame, color: str, label: str, **line: Any) -> None:
    """Media (línea) ± desvío (banda) de curve_stats."""
    ax.plot(stats["epoch"], stats["mean"], color=color, label=label, **line)
    std = stats["std"].fillna(0.0)
    ax.fill_between(
        stats["epoch"], stats["mean"] - std, stats["mean"] + std, color=color, alpha=0.18, lw=0
    )


def _log_band(ax: Any, log_stats: pd.DataFrame, color: str, label: str) -> None:
    """Curva en escala log: 10^(media ± desvío) de log₁₀ del valor (curve_stats con log10).

    En escala log el desvío de los valores crudos suele ser del orden de la
    media y la banda media − desvío caería a ≤ 0; en log₁₀ es simétrica.
    """
    mean, std = log_stats["mean"], log_stats["std"].fillna(0.0)
    ax.plot(log_stats["epoch"], 10**mean, color=color, label=label)
    ax.fill_between(
        log_stats["epoch"], 10 ** (mean - std), 10 ** (mean + std), color=color, alpha=0.18, lw=0
    )


def _log10(values: pd.Series) -> pd.Series:
    return np.log10(np.maximum(values, np.finfo(float).tiny))


def _logic_points(ax: Any, X: np.ndarray, y: np.ndarray) -> None:
    """Los 4 puntos de AND/XOR: y = +1 círculo lleno, y = −1 círculo vacío."""
    positive = y[:, 0] > 0
    style = {"s": 130, "edgecolors": _INK, "linewidths": 1.8, "zorder": 5}
    ax.scatter(*X[positive].T, color=_INK, label="y = +1", **style)
    ax.scatter(*X[~positive].T, color="white", label="y = −1", **style)


def _square_axes(ax: Any) -> None:
    ax.set_xlim(-_LIM, _LIM)
    ax.set_ylim(-_LIM, _LIM)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_aspect("equal")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")


def _save(fig: Any, out_dir: Path, name: str) -> None:
    assert name in FIGURES, name
    save_figure(fig, out_dir / name)


# --- V-01: AND ---


def _weights_history(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(épocas (n,), W (n, 2), b (n,)) del weights_history.npz de un perceptrón [2, 1]."""
    path = run_dir / "weights_history.npz"
    if not path.exists():
        raise MissingResults(
            f"{path}: falta el historial de pesos (and_step.json necesita "
            "logging.save_weights_history: true; relanzar con --force)"
        )
    with np.load(path) as history:
        return history["epoch"], history["W_0"][:, :, 0], history["b_0"][:, 0, 0]


def plot_and(results_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Recta de decisión por época (la semilla que más tarda en converger) y error vs época.

    history.csv empieza en la época 1; la época 0 (pesos iniciales) sale de
    weights_history.npz, prediciendo con el escalón sobre los 4 puntos.
    """
    import matplotlib.pyplot as plt

    runs, path = validation_runs(results_dir, "and_step")
    lr = float(runs["training.optimizer.lr"].iloc[0])
    n = len(runs)
    X, y = run_dataset(path / runs["run"].iloc[0])

    initial = []
    for run in runs["run"]:
        _, W, b = _weights_history(path / run)
        y_pred = Step().forward(X @ W[0].reshape(-1, 1) + b[0])
        initial.append(
            {"run": run, "epoch": 0, "metric": "train_loss", "value": error_rate(y, y_pred)}
        )
    histories = pd.concat([pd.DataFrame(initial), load_histories(path)], ignore_index=True)
    convergence = run_convergence(histories, "train_loss")

    # Recta de decisión por época de la semilla que más tarda (empate: la menor).
    slowest = max(runs["run"], key=lambda r: np.inf if convergence[r] is None else convergence[r])
    row = runs[runs["run"] == slowest].iloc[0]
    epochs, W, b = _weights_history(path / slowest)
    last = convergence[slowest]
    last = int(epochs[-1]) if last is None else last
    shown = np.flatnonzero(epochs <= last)
    if shown.size > _MAX_LINES:
        shown = shown[np.unique(np.linspace(0, shown.size - 1, _MAX_LINES).round().astype(int))]
    ramp = [_BLUE_RAMP[i] for i in np.linspace(1, len(_BLUE_RAMP) - 1, shown.size).astype(int)]

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for color, i in zip(ramp, shown, strict=True):
        line = decision_line(W[i], b[i], _LIM)
        if line is None:
            continue
        final = epochs[i] == last
        if epochs[i] == 0:
            label = "época 0 (pesos iniciales)"
        elif final:
            label = f"época {epochs[i]}: error 0"
        else:
            label = f"época {epochs[i]}"
        ax.plot(*line, color=color, lw=3.0 if final else 1.8, label=label)
    _logic_points(ax, X, y)
    _square_axes(ax)
    ax.set_title(
        "AND con perceptrón escalón: recta de decisión por época\n"
        f"(semilla {row['seed']}, la que más épocas necesita; η = {lr:g})"
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    _save(fig, out_dir, "V-01_and_decision_boundary")

    # Tasa de error vs época, todas las semillas.
    converged = [int(e) for e in convergence if e is not None]
    stats = curve_stats(histories, "train_loss")
    fig, ax = plt.subplots()
    _band(ax, stats, PALETTE[0], "media ± desvío")
    xmax = max(converged) + 3 if len(converged) == n else int(stats["epoch"].max())
    ax.set_xlim(0, xmax)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("época (0 = pesos iniciales)")
    ax.set_ylabel("tasa de error (fracción mal clasificada)")
    ax.set_title(f"AND con escalón: error de entrenamiento (media ± desvío, n = {n} semillas)")
    conv = describe(converged)
    ax.text(
        0.97,
        0.95,
        f"error 0 en {len(converged)}/{n} semillas\n"
        f"época de convergencia: {_pm(conv, '.2g')}\n(mín {conv['min']:g}, máx {conv['max']:g})"
        if converged
        else f"ninguna semilla llega a error 0 (n = {n})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=_INK_SECONDARY,
    )
    _save(fig, out_dir, "V-01_and_error")

    return {
        "run_name": config_run_name("and_step"),
        "n_seeds": n,
        "lr": lr,
        "epochs": int(runs["training.epochs"].iloc[0]),
        "converged": len(converged),
        "convergence_epoch": conv,
        "decision_boundary_seed": int(row["seed"]),
    }


# --- V-02 y V-03: rectas ---


def plot_line_fit(
    results_dir: Path, out_dir: Path, config: str, name: str, what: str
) -> dict[str, Any]:
    """Datos + curva ajustada (media ± desvío entre semillas) y MSE vs época."""
    import matplotlib.pyplot as plt

    runs, path = validation_runs(results_dir, config)
    n = len(runs)
    X, y = run_dataset(path / runs["run"].iloc[0])
    grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    fits, weights, biases = [], [], []
    for run in runs["run"]:
        y_grid, network = predict_model(path / run / "model.npz", grid)
        fits.append(y_grid[:, 0])
        weights.append(float(network.layers[0].W[0, 0]))
        biases.append(float(network.layers[0].b[0, 0]))
    fits = np.array(fits)
    mse = describe(runs["train.mse"])
    w, b = describe(weights), describe(biases)

    fig, (ax_fit, ax_loss) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax_fit.scatter(
        X[:, 0], y[:, 0], s=45, color="white", edgecolors=_INK_SECONDARY, lw=1.3,
        label=f"datos (n = {len(y)})", zorder=3,
    )  # fmt: skip
    mean = fits.mean(axis=0)
    std = fits.std(axis=0, ddof=1) if n > 1 else np.zeros(len(grid))
    ax_fit.plot(grid[:, 0], mean, color=PALETTE[0], label=f"red (media de {n} semillas)")
    ax_fit.fill_between(grid[:, 0], mean - std, mean + std, color=PALETTE[0], alpha=0.25, lw=0)
    ax_fit.set_xlabel("x")
    ax_fit.set_ylabel("y")
    if runs["model.output_activation"].iloc[0] in ("tanh", "sigmoid"):
        what += f" (β = {runs['model.beta'].iloc[0]:g})"
    ax_fit.set_title(what)
    ax_fit.legend(loc="upper left")
    ax_fit.text(
        0.97,
        0.05,
        f"MSE final = {_pm(mse, '.2e')}\nw = {_pm(w, '.4f')}\nb = {_pm(b, '.4f')}",
        transform=ax_fit.transAxes,
        ha="right",
        va="bottom",
        color=_INK_SECONDARY,
    )

    lr = float(runs["training.optimizer.lr"].iloc[0])
    log_stats = curve_stats(load_histories(path), "train_loss", transform=_log10)
    _log_band(ax_loss, log_stats, PALETTE[0], "media ± desvío de log₁₀")
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("época")
    ax_loss.set_ylabel("MSE de entrenamiento (escala log)")
    ax_loss.set_title(f"MSE vs época (η = {lr:g}, n = {n} semillas)")
    ax_loss.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, out_dir, name)

    return {
        "run_name": config_run_name(config),
        "n_seeds": n,
        "lr": lr,
        "epochs": int(runs["training.epochs"].iloc[0]),
        "train_mse": mse,
        "w": w,
        "b": b,
    }


# --- V-04: XOR ---


def _xor_group(runs: pd.DataFrame, histories: pd.DataFrame) -> dict[str, Any]:
    solved = runs["train.accuracy"] == 1.0
    to_solve = run_convergence(histories, "train_accuracy", error=False)
    return {
        "run_name": runs["run_name"].iloc[0],
        "n_seeds": len(runs),
        "solved": int(solved.sum()),
        "solved_seeds": [int(s) for s in runs.loc[solved, "seed"]],
        "unsolved_seeds": [int(s) for s in runs.loc[~solved, "seed"]],
        "epochs_to_solve": describe([e for e in to_solve if e is not None]),
        "final_loss": describe(runs["train.loss"]),
    }


def _regions(ax: Any, model: Path, X: np.ndarray, y: np.ndarray, title: str) -> Any:
    from matplotlib.colors import LinearSegmentedColormap

    ticks = np.linspace(-_LIM, _LIM, 300)
    xx, yy = np.meshgrid(ticks, ticks)
    out, _ = predict_model(model, np.column_stack([xx.ravel(), yy.ravel()]))
    out = out[:, 0].reshape(xx.shape)
    cmap = LinearSegmentedColormap.from_list("xor", _DIVERGING)
    filled = ax.contourf(xx, yy, out, levels=np.linspace(-1, 1, 21), cmap=cmap, alpha=0.75)
    ax.contour(xx, yy, out, levels=[0.0], colors=_INK, linewidths=2.0)
    _logic_points(ax, X, y)
    _square_axes(ax)
    ax.grid(False)
    ax.set_title(title, fontsize=14)
    return filled


def plot_xor(results_dir: Path, out_dir: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    step, step_path = validation_runs(results_dir, "xor_step")
    mlps = {c: validation_runs(results_dir, c) for c in ("xor_221", "xor_2321")}
    step_hist = load_histories(step_path)
    mlp_hist = {c: load_histories(path) for c, (_, path) in mlps.items()}

    # Error vs época: escalón (train_loss = tasa de error) vs MLP (1 − accuracy).
    # Colores fijos por modelo en todas las figuras: [2,2,1], [2,3,2,1], escalón.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for color, (config, (runs, _)) in zip(PALETTE, mlps.items(), strict=False):
        stats = curve_stats(mlp_hist[config], "train_accuracy", transform=lambda v: 1 - v)
        _band(ax, stats, color, f"MLP {runs['model.layers'].iloc[0]} (n = {len(runs)})")
    _band(ax, curve_stats(step_hist, "train_loss"), PALETTE[2], f"escalón [2, 1] (n = {len(step)})")
    epochs = int(step["training.epochs"].iloc[0])
    ax.axhline(0.25, color=_INK_SECONDARY, lw=1.0, ls=(0, (4, 3)))
    ax.text(
        epochs / 1.15, 0.265, "mínimo posible con una recta: 1/4",
        color=_INK_SECONDARY, ha="right", va="bottom",
    )  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlim(1, epochs)
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlabel("época (escala log)")
    ax.set_ylabel("tasa de error de entrenamiento")
    ax.set_title("XOR: error vs época, escalón vs MLP (media ± desvío)")
    ax.legend(loc="upper right")
    _save(fig, out_dir, "V-04_xor_error_step_vs_mlp")

    # Regiones de decisión: escalón, [2,2,1] que resuelve y peor, [2,3,2,1].
    X, y = run_dataset(step_path / step["run"].iloc[0])

    def pick(runs: pd.DataFrame, worst: bool = False) -> pd.Series:
        if worst:
            return runs.loc[runs["train.loss"].idxmax()]
        solved = runs[runs["train.accuracy"] == 1.0]
        return solved.iloc[0] if len(solved) else runs.loc[runs["train.loss"].idxmin()]

    def verdict(row: pd.Series) -> str:
        return "resuelve" if row["train.accuracy"] == 1.0 else "no resuelve"

    (runs_221, path_221), (runs_2321, path_2321) = mlps["xor_221"], mlps["xor_2321"]
    panels = [
        (step_path, step.iloc[0], "escalón [2, 1]"),
        (path_221, pick(runs_221), "MLP [2, 2, 1]"),
        (path_221, pick(runs_221, worst=True), "MLP [2, 2, 1], peor semilla"),
        (path_2321, pick(runs_2321), "MLP [2, 3, 2, 1]"),
    ]
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 11.5), sharex=True, sharey=True, layout="constrained"
    )
    for ax, (path, row, name) in zip(axes.ravel(), panels, strict=True):
        title = f"{name}\nsemilla {row['seed']}: {verdict(row)}"
        filled = _regions(ax, path / row["run"] / "model.npz", X, y, title)
        ax.label_outer()
    fig.colorbar(filled, ax=axes, shrink=0.6, ticks=[-1, 0, 1], label="salida de la red")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.suptitle("XOR: regiones de decisión (línea negra: salida = 0)")
    _save(fig, out_dir, "V-04_xor_decision_regions")

    step_errors = step_hist[step_hist["metric"] == "train_loss"]
    final_step = step_errors[step_errors["epoch"] == step_errors["epoch"].max()]["value"]
    return {
        "step": {
            "run_name": config_run_name("xor_step"),
            "n_seeds": len(step),
            "epochs": int(step["training.epochs"].iloc[0]),
            "min_error": float(step_errors["value"].min()),
            "final_error": describe(final_step),
            "converged": int(run_convergence(step_hist, "train_loss").notna().sum()),
        },
        "mlp_221": _xor_group(runs_221, mlp_hist["xor_221"]),
        "mlp_2321": _xor_group(runs_2321, mlp_hist["xor_2321"]),
    }


# --- T6: escala de inicialización en XOR ---


def plot_xor_init(results_dir: Path, out_dir: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    runs = init_runs(results_dir)
    runs["solved"] = runs["train.accuracy"] == 1.0
    inits = [i for i in _INIT_ORDER if i in set(runs["init"])]
    inits += sorted(set(runs["init"]) - set(inits))
    archs = [a for a in _ARCH_ORDER if a in set(runs["model.layers"])]
    epochs = int(runs["training.epochs"].iloc[0])
    lr = float(runs["training.optimizer.lr"].iloc[0])
    n = int(runs.groupby(["model.layers", "init"]).size().iloc[0])

    # % de semillas que resuelven, por inicialización y arquitectura.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.38
    x = np.arange(len(inits))
    for k, arch in enumerate(archs):
        group = runs[runs["model.layers"] == arch].groupby("init")["solved"]
        solved = group.sum().reindex(inits)
        total = group.size().reindex(inits)
        pct = 100 * solved / total
        offset = (k - (len(archs) - 1) / 2) * (width + 0.02)
        bars = ax.bar(x + offset, pct, width, color=PALETTE[k], label=f"MLP {arch}")
        for bar, s, t in zip(bars, solved, total, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{int(s)}/{int(t)}",
                ha="center",
                va="bottom",
                color=_INK_SECONDARY,
                fontsize=12,
            )
    ax.set_xticks(x, inits)
    ax.set_ylim(0, 124)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("inicialización de los pesos")
    ax.set_ylabel("semillas que resuelven XOR (%)")
    ax.set_title(
        f"XOR: semillas con accuracy 100 % tras {epochs} épocas (η = {lr:g}, n = {n} por barra)"
    )
    ax.legend(loc="upper left", ncol=len(archs))
    ax.grid(axis="x", visible=False)
    _save(fig, out_dir, "V-04_xor_init_solved")

    # Loss vs época por inicialización, un panel por arquitectura.
    histories = []
    for run_name in runs["run_name"].unique():
        long = load_histories(Path(results_dir) / run_name)
        long = long[long["metric"] == "train_loss"].assign(run_name=run_name)
        histories.append(long)
    histories = pd.concat(histories, ignore_index=True).merge(
        runs[["run_name", "run", "init", "model.layers"]], on=["run_name", "run"]
    )
    fig, axes = plt.subplots(1, len(archs), figsize=(14, 5.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, arch in zip(axes, archs, strict=True):
        for color, init in zip(PALETTE, inits, strict=False):
            group = histories[(histories["model.layers"] == arch) & (histories["init"] == init)]
            highlight = init == inits[0]
            _band(ax, curve_stats(group, "train_loss"), color, init, lw=3.0 if highlight else 1.8)
        # Salida ≈ 0 con targets ±1: MSE = 1. Es donde arrancan los pesos chicos.
        ax.text(
            epochs * 0.275, 0.97, "meseta: salida ≈ 0, MSE ≈ 1",
            ha="center", va="top", color=_INK_SECONDARY,
        )  # fmt: skip
        ax.set_xlabel("época")
        ax.set_title(f"MLP {arch}")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("MSE de entrenamiento")
    axes[-1].legend(title="inicialización", loc="upper right")
    fig.suptitle(
        f"XOR: loss vs época según la inicialización (η = {lr:g}, media ± desvío, n = {n} semillas)"
    )
    fig.tight_layout()
    _save(fig, out_dir, "V-04_xor_init_loss")

    by_init: dict[str, Any] = {}
    for init in inits:
        by_init[init] = {}
        for arch in archs:
            group = runs[(runs["init"] == init) & (runs["model.layers"] == arch)]
            by_init[init][arch] = {
                "run_name": group["run_name"].iloc[0],
                "n": len(group),
                "solved": int(group["solved"].sum()),
                "solved_seeds": [int(s) for s in group.loc[group["solved"], "seed"]],
                "final_loss": describe(group["train.loss"]),
            }
    return {"epochs": epochs, "lr": lr, "by_init": by_init}


# --- CLI ---


def _report(summary: Mapping[str, Any]) -> list[str]:
    """Los números clave de la slide, en texto."""
    and_, xor = summary["V-01"], summary["V-04"]
    lines = [
        f"V-01 AND: error 0 en {and_['converged']}/{and_['n_seeds']} semillas; época de "
        f"convergencia {_pm(and_['convergence_epoch'], '.3g')} "
        f"(mín {and_['convergence_epoch']['min']:g}, máx {and_['convergence_epoch']['max']:g})",
    ]
    for key, what in (("V-02", "y = x"), ("V-03", "y = tanh(x)")):
        fit = summary[key]
        lines.append(
            f"{key} {what}: MSE final {_pm(fit['train_mse'], '.2e')}; "
            f"w = {_pm(fit['w'], '.4f')}, b = {_pm(fit['b'], '.4f')} (n = {fit['n_seeds']})"
        )
    lines.append(
        f"V-04 escalón en XOR: error mínimo {xor['step']['min_error']:.2f}, "
        f"{xor['step']['converged']}/{xor['step']['n_seeds']} semillas llegan a error 0"
    )
    for key in ("mlp_221", "mlp_2321"):
        group = xor[key]
        lines.append(
            f"V-04 {group['run_name']}: resuelven {group['solved']}/{group['n_seeds']} "
            f"(no resuelven: {group['unsolved_seeds']}); épocas hasta 100 %: "
            f"{_pm(group['epochs_to_solve'], '.3g')}"
        )
    for label, by_arch in summary["V-04_init"]["by_init"].items():
        cells = ", ".join(f"{arch}: {g['solved']}/{g['n']}" for arch, g in by_arch.items())
        lines.append(f"V-04 init {label}: {cells}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Genera todas las figuras y resumen.json. Devuelve 0, o 2 si faltan corridas."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(
        prog="python -m analysis.validacion",
        description="Figuras de validación del motor (V-01 a V-04) desde results/.",
    )
    parser.add_argument("--results-dir", default="results", help="carpeta raíz de resultados")
    parser.add_argument("--out-dir", default="figures/validacion", help="carpeta de las figuras")
    args = parser.parse_args(argv)
    results_dir, out_dir = Path(args.results_dir), Path(args.out_dir)

    import matplotlib

    # Solo se guardan archivos: el backend sin ventanas evita depender de Tk
    # (en Windows su instalación puede fallar al abrir una figura).
    matplotlib.use("Agg")
    try:
        check_results(results_dir)
        apply_style()
        summary = {
            "V-01": plot_and(results_dir, out_dir),
            "V-02": plot_line_fit(
                results_dir,
                out_dir,
                "linear_identity",
                "V-02_linear_fit",
                "y = x con perceptrón lineal",
            ),
            "V-03": plot_line_fit(
                results_dir,
                out_dir,
                "nonlinear_tanh",
                "V-03_tanh_fit",
                "y = tanh(x) con perceptrón no lineal tanh",
            ),
            "V-04": plot_xor(results_dir, out_dir),
            "V-04_init": plot_xor_init(results_dir, out_dir),
        }
    except (MissingResults, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    with open(out_dir / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    for line in _report(summary):
        print(line)
    print(f"{len(FIGURES)} figuras (.png y .pdf) y resumen.json en {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
