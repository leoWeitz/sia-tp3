"""Ej1 · aprendizaje del perceptrón lineal vs no lineal (F10): figuras 1–6 y tabla final.

    python -m analysis.ej1_learning [--results-dir results] [--out-dir figures/ej1/learning]

Solo lee results/ (CLAUDE.md §2.6). Antes hay que correr, con el runner:

    python -m experiments.runner experiments/configs/ej1/<learning|long|beta|scaling>.json

Referencias (se calculan acá, no en el runner, y no son modelos del TP):
- predictor constante: predecir la media del target → MSE = varianza del target.
- mínimos cuadrados: np.linalg.lstsq con bias sobre las mismas features. Es la
  cota exacta del perceptrón lineal con MSE: si la alcanza, agotó su capacidad.

Saturación de la neurona: con la logística de la cátedra s = θ(h) =
1 / (1 + e^{−2βh}), θ'(h) = 2β·s(1 − s), cuyo máximo es β/2. "θ' < 5 % del
máximo" equivale a 4·s(1 − s) < 0.05, así que se mide sobre las salidas
guardadas (predictions.npz) sin volver a cargar el modelo.

Meseta: mejora relativa del MSE de entrenamiento < 0.1 % en el último 10 % de
las épocas (docs/fases/F10).

Figuras (FIGURES), en <out-dir> como .png y .pdf, más tabla.csv y resumen.json:
- E1-A-a_train_mse: MSE de entrenamiento vs época (log), mejor η de cada modelo,
  media ± desvío entre semillas, con las dos referencias.
- E1-A_lr_sensitivity: MSE final vs η por modelo y modo (online / batch).
- E1-A-b_plateau: ej1_long (Adam, 5× épocas) vs la mejor curva de ej1_learning.
- E1-A-b_linear_outputs: histograma de salidas del lineal con [0, 1] marcado.
- E1-A-b_saturation_beta: fracción de neurona saturada y MSE final vs β, con y
  sin normalización (ej1_beta).
- E1-A-b_scaling: MSE final por normalización y modelo (ej1_scaling).
- E1-A-c_summary_table: tabla final (MSE ± desvío, mejora vs constante,
  distancia a mínimos cuadrados, % fuera de [0, 1], s/época, métricas
  secundarias contra flagged_fraud con umbral 0.5).
"""

import argparse
import json
import sys
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.common import PALETTE, apply_style, load_histories, load_runs, save_figure
from core.metrics import accuracy, f1, precision, recall
from experiments.config import load_run_config
from experiments.runner import load_dataset

EXPERIMENTS = ("ej1_learning", "ej1_long", "ej1_beta", "ej1_scaling")
LABEL = "flagged_fraud"
MODEL_NAMES = {"identity": "lineal", "sigmoid": "logística"}
MODEL_COLORS = {"identity": PALETTE[0], "sigmoid": PALETTE[1]}
SATURATION_FRACTION = 0.05
PLATEAU_TAIL = 0.10
PLATEAU_REL = 1e-3
SECONDARY_THRESHOLD = 0.5
FIGURES = (
    "E1-A-a_train_mse",
    "E1-A_lr_sensitivity",
    "E1-A-b_plateau",
    "E1-A-b_linear_outputs",
    "E1-A-b_saturation_beta",
    "E1-A-b_scaling",
    "E1-A-c_summary_table",
)


# --- Referencias y cálculos ---


def constant_mse(y: np.ndarray) -> float:
    """MSE del predictor constante óptimo (la media): la varianza poblacional de y."""
    y = np.asarray(y, dtype=np.float64).ravel()
    return float(np.mean((y - y.mean()) ** 2))


def least_squares_mse(X: np.ndarray, y: np.ndarray) -> float:
    """MSE de la solución de mínimos cuadrados con bias (cota del perceptrón lineal).

    X: (p, n) crudo · y: (p,) o (p, 1). Se estandariza X antes de resolver solo
    por condicionamiento numérico: con bias, la solución es invariante a
    transformaciones afines de cada feature, así que el MSE no cambia.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(len(X), -1)
    std = X.std(axis=0)
    Xs = (X - X.mean(axis=0)) / np.where(std > 0, std, 1.0)
    A = np.hstack([Xs, np.ones((len(X), 1))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.mean((A @ coef - y) ** 2))


def plateau(losses: np.ndarray, tail: float = PLATEAU_TAIL, tol: float = PLATEAU_REL) -> dict:
    """Mejora relativa del error en el último `tail` de las épocas y si es una meseta.

    losses: (épocas,). rel = (L[inicio del tramo] − L[final]) / L[inicio del tramo];
    meseta si rel < tol.
    """
    losses = np.asarray(losses, dtype=np.float64)
    n_tail = max(int(np.ceil(tail * len(losses))), 1)
    start, end = losses[-n_tail - 1] if n_tail < len(losses) else losses[0], losses[-1]
    rel = float((start - end) / start) if start != 0 else 0.0
    return {"mejora_relativa": rel, "meseta": bool(rel < tol), "epocas_tramo": n_tail}


def saturated_fraction(outputs: np.ndarray, fraction: float = SATURATION_FRACTION) -> float:
    """Fracción de salidas s de la logística con θ'(h) < fraction · máx θ' (4s(1−s) < fraction)."""
    s = np.asarray(outputs, dtype=np.float64).ravel()
    return float(np.mean(4.0 * s * (1.0 - s) < fraction))


def out_of_range_fraction(outputs: np.ndarray) -> float:
    """Fracción de salidas fuera de [0, 1] (no interpretables como probabilidad)."""
    o = np.asarray(outputs, dtype=np.float64).ravel()
    return float(np.mean((o < 0.0) | (o > 1.0)))


def mse_by_class(labels: np.ndarray, y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """MSE contra el target separado por flagged_fraud: dónde está el error que queda.

    labels, y, scores: (p,). Devuelve mse_negativos, mse_positivos y la fracción
    del error total (suma de cuadrados) que aportan los positivos.
    """
    lab = np.asarray(labels).astype(int).ravel()
    sq = (
        np.asarray(scores, dtype=np.float64).ravel() - np.asarray(y, dtype=np.float64).ravel()
    ) ** 2
    return {
        "mse_negativos": float(sq[lab == 0].mean()),
        "mse_positivos": float(sq[lab == 1].mean()),
        "fraccion_error_positivos": float(sq[lab == 1].sum() / sq.sum()),
    }


def label_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Accuracy, precision, recall y F1 contra flagged_fraud con score ≥ threshold (secundario)."""
    y = np.asarray(labels).astype(int).ravel()
    pred = (np.asarray(scores).ravel() >= threshold).astype(int)
    return {
        "accuracy": accuracy(y, pred),
        "precision": precision(y, pred),
        "recall": recall(y, pred),
        "f1": f1(y, pred),
    }


def model_label(row: pd.Series) -> str:
    """'lineal' / 'logística' según model.output_activation."""
    return MODEL_NAMES[row["model.output_activation"]]


def mode_label(batch_size: Any) -> str:
    """'online' (batch 1), 'batch' (lote completo) o 'mini-batch <n>'."""
    if batch_size is None or (isinstance(batch_size, float) and np.isnan(batch_size)):
        return "batch"
    return "online" if int(batch_size) == 1 else f"mini-batch {int(batch_size)}"


def per_config(runs: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Una fila por hash: claves, n, n_ok y MSE final (media, desvío) de las corridas ok.

    Una config con alguna semilla divergida queda con n_ok < n y no se elige como mejor.
    """
    rows = []
    for h, g in runs.groupby("hash", sort=True):
        ok = g[g["status"] == "ok"]
        row = {k: g[k].iloc[0] for k in keys}
        row.update(
            {
                "hash": h,
                "n": len(g),
                "n_ok": len(ok),
                "mse_mean": ok["train.loss"].mean() if len(ok) else np.nan,
                "mse_std": ok["train.loss"].std(ddof=1) if len(ok) > 1 else np.nan,
                "s_per_epoch": g["s_per_epoch"].mean(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def best_by_model(table: pd.DataFrame) -> dict[str, pd.Series]:
    """Config de menor MSE medio por modelo, entre las que convergieron en todas las semillas."""
    out = {}
    for act, g in table.groupby("model.output_activation"):
        full = g[(g["n_ok"] == g["n"]) & g["mse_mean"].notna()]
        if full.empty:
            raise ValueError(
                f"Ninguna config de {MODEL_NAMES[act]} convergió en todas las semillas"
            )
        out[act] = full.loc[full["mse_mean"].idxmin()]
    return out


def curve(histories: pd.DataFrame, hash_: str, metric: str = "train_loss") -> pd.DataFrame:
    """Media, desvío (ddof = 1) y n por época de una config (hash)."""
    rows = histories[(histories["hash"] == hash_) & (histories["metric"] == metric)]
    if rows.empty:
        raise ValueError(f"No hay {metric!r} para la config {hash_}")
    g = rows.groupby("epoch")["value"]
    return pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=1), "n": g.size()}).reset_index()


def run_outputs(results_dir: Path, runs: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Salidas de entrenamiento (predictions.npz) de cada corrida: run → {idx, y_score}."""
    out = {}
    for run in runs["run"]:
        z = np.load(results_dir / run / "predictions.npz")
        out[run] = {"idx": z["idx"], "y_score": z["y_score"].ravel()}
    return out


def dataset_arrays(run_dir: str | PathLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X crudo (p, n), y (p,), flagged_fraud (p,)) del dataset de una corrida.

    La etiqueta se lee del mismo CSV solo para las métricas secundarias: no
    participa del entrenamiento (está en dataset.drop).
    """
    config = load_run_config(Path(run_dir) / "config.json")
    data = load_dataset(config["dataset"])
    labels = pd.read_csv(config["dataset"]["path"], usecols=[LABEL])[LABEL].to_numpy()
    return data.X, np.asarray(data.y, dtype=np.float64).ravel(), labels


# --- Figuras ---


def _band(ax: Any, c: pd.DataFrame, color: str, label: str) -> None:
    ax.plot(c["epoch"], c["mean"], color=color, label=label)
    std = c["std"].fillna(0.0)
    ax.fill_between(c["epoch"], c["mean"] - std, c["mean"] + std, color=color, alpha=0.2, lw=0)


def _references(ax: Any, refs: dict[str, float]) -> None:
    ax.axhline(refs["constante"], color="#52514e", ls="--", lw=1.5, label="predictor constante")
    ax.axhline(
        refs["minimos_cuadrados"], color="#52514e", ls=":", lw=1.5, label="mínimos cuadrados"
    )


def fig_train_mse(histories, best, refs) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for act, row in best.items():
        c = curve(histories, row["hash"])
        label = (
            f"{MODEL_NAMES[act]} (η = {row['training.optimizer.lr']:g}, "
            f"{mode_label(row['training.batch_size'])}, n = {int(c['n'].min())})"
        )
        _band(ax, c, MODEL_COLORS[act], label)
    _references(ax, refs)
    ax.set_yscale("log")
    ax.set_xlabel("época")
    ax.set_ylabel("MSE de entrenamiento")
    ax.set_title("Lineal vs logística · todas las muestras · media ± desvío")
    ax.legend(fontsize=11)
    return fig


def fig_lr_sensitivity(table, refs) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    styles = {"online": "-", "batch": "--"}
    for (act, mode), g in table.groupby(["model.output_activation", "modo"]):
        g = g.sort_values("training.optimizer.lr")
        ok = g["n_ok"] == g["n"]
        ax.errorbar(
            g.loc[ok, "training.optimizer.lr"],
            g.loc[ok, "mse_mean"],
            yerr=g.loc[ok, "mse_std"].fillna(0.0),
            color=MODEL_COLORS[act],
            ls=styles.get(mode, "-"),
            marker="o",
            capsize=3,
            label=f"{MODEL_NAMES[act]} · {mode}",
        )
        for _, r in g[~ok].iterrows():
            ax.annotate(
                f"divergió ({r['n'] - r['n_ok']}/{r['n']})",
                (r["training.optimizer.lr"], refs["constante"]),
                color=MODEL_COLORS[act],
                fontsize=10,
                ha="center",
                rotation=90,
                va="bottom",
            )
    _references(ax, refs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("tasa de aprendizaje η")
    ax.set_ylabel("MSE de entrenamiento final")
    ax.set_title("Sensibilidad a η (500 épocas, media ± desvío)")
    ax.legend(fontsize=11)
    return fig


def fig_plateau(hist_learning, hist_long, best, long_best, refs, plateaus) -> Any:
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for act in best:
        c_l = curve(hist_learning, best[act]["hash"])
        c_g = curve(hist_long, long_best[act]["hash"])
        p = plateaus[act]
        _band(a1, c_l, MODEL_COLORS[act], f"{MODEL_NAMES[act]} · GD (ej1_learning)")
        a1.plot(
            c_g["epoch"],
            c_g["mean"],
            color=MODEL_COLORS[act],
            ls="--",
            label=f"{MODEL_NAMES[act]} · Adam (ej1_long)",
        )
        # Zoom: exceso de cada época sobre el MSE final del propio modelo, en %.
        tail = c_g[c_g["epoch"] > c_g["epoch"].max() * (1 - 2 * PLATEAU_TAIL)]
        final = tail["mean"].iloc[-1]
        a2.plot(
            tail["epoch"],
            100 * (tail["mean"] / final - 1),
            color=MODEL_COLORS[act],
            label=f"{MODEL_NAMES[act]} · mejora último 10 %: {100 * p['mejora_relativa']:.3f} %",
        )
    a1.axhline(
        refs["minimos_cuadrados"], color="#52514e", ls=":", lw=1.5, label="mínimos cuadrados"
    )
    a1.axhline(refs["constante"], color="#52514e", ls="--", lw=1.5, label="predictor constante")
    a1.set_yscale("log")
    a1.set_ylabel("MSE de entrenamiento")
    a1.set_title("Más épocas y otro optimizador")
    a2.axhline(0, color="#52514e", lw=1)
    a2.axvline(
        tail["epoch"].max() * (1 - PLATEAU_TAIL),
        color="#52514e",
        ls=":",
        lw=1.5,
        label="inicio del último 10 %",
    )
    a2.set_ylabel("MSE / MSE final − 1 (%)")
    a2.set_title("Zoom: último 20 % de ej1_long (Adam)")
    for ax in (a1, a2):
        ax.set_xlabel("época")
    a1.legend(fontsize=10)
    a2.legend(fontsize=10)
    fig.tight_layout()
    return fig


def fig_linear_outputs(scores: np.ndarray, frac_mean: float, frac_std: float, label: str) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axvspan(0, 1, color="#e1e0d9", alpha=0.6, lw=0, label="[0, 1]")
    ax.hist(scores, bins=60, color=MODEL_COLORS["identity"], label="salidas del lineal")
    ax.set_xlabel("salida del perceptrón lineal")
    ax.set_ylabel("muestras (todas las semillas)")
    ax.set_title(f"{label} · fuera de [0, 1]: {100 * frac_mean:.1f} % ± {100 * frac_std:.1f} %")
    ax.legend()
    return fig


def fig_saturation_beta(beta_table: pd.DataFrame) -> Any:
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for i, (norm, g) in enumerate(beta_table.groupby("dataset.normalize")):
        g = g.sort_values("model.beta")
        color = PALETTE[2 * i]
        label = "sin normalizar" if norm == "none" else norm
        a1.errorbar(
            g["model.beta"],
            100 * g["sat_mean"],
            yerr=100 * g["sat_std"].fillna(0),
            marker="o",
            capsize=3,
            color=color,
            label=label,
        )
        ok = g["n_ok"] == g["n"]
        a2.errorbar(
            g.loc[ok, "model.beta"],
            g.loc[ok, "mse_mean"],
            yerr=g.loc[ok, "mse_std"].fillna(0),
            marker="o",
            capsize=3,
            color=color,
            label=label,
        )
    a1.set_ylabel(f"% de muestras con θ' < {100 * SATURATION_FRACTION:.0f} % del máximo")
    a1.set_title("Neurona saturada vs β")
    a2.set_yscale("log")
    a2.set_ylabel("MSE de entrenamiento final")
    a2.set_title("Error final vs β")
    for ax in (a1, a2):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("β")
        ax.legend()
    fig.suptitle("Logística · media ± desvío entre semillas")
    fig.tight_layout()
    return fig


def fig_scaling(scaling_table: pd.DataFrame, refs: dict[str, float]) -> Any:
    import matplotlib.pyplot as plt

    order = ["none", "minmax", "zscore"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for j, (act, g) in enumerate(scaling_table.groupby("model.output_activation")):
        g = g.set_index("dataset.normalize").reindex(order)
        x = np.arange(len(order)) + (j - 0.5) * 0.3
        ok = (g["n_ok"] == g["n"]).to_numpy()
        ax.bar(
            x[ok],
            g["mse_mean"].to_numpy()[ok],
            width=0.28,
            color=MODEL_COLORS[act],
            yerr=g["mse_std"].fillna(0).to_numpy()[ok],
            capsize=3,
            label=MODEL_NAMES[act],
        )
        for xi, (_, r) in zip(x[~ok], g[~ok].iterrows(), strict=True):
            ax.text(
                xi,
                refs["minimos_cuadrados"],
                f"divergió\n({int(r['n'] - r['n_ok'])}/{int(r['n'])})",
                ha="center",
                va="bottom",
                fontsize=10,
                color=MODEL_COLORS[act],
            )
    _references(ax, refs)
    ax.set_xticks(range(len(order)), ["sin normalizar", "min-max", "z-score"])
    ax.set_yscale("log")
    ax.set_ylabel("MSE de entrenamiento final")
    ax.set_title("Normalización de las features (GD batch, mismo η)")
    ax.legend(fontsize=11)
    return fig


def fig_summary_table(table: pd.DataFrame) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(20, 1.2 + 0.6 * len(table)))
    ax.axis("off")
    tab = ax.table(cellText=table.to_numpy(), colLabels=list(table.columns), loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.auto_set_column_width(list(range(len(table.columns))))
    tab.scale(1, 1.7)
    for (r, _), cell in tab.get_celld().items():
        cell.set_edgecolor("#e1e0d9")
        if r == 0:
            cell.set_text_props(weight="bold")
    ax.set_title("Ej1 · aprendizaje con todas las muestras (media ± desvío entre semillas)")
    return fig


# --- Armado ---


def _pm(mean: float, std: float, fmt: str = "{:.4g}") -> str:
    return f"{fmt.format(mean)} ± {fmt.format(0.0 if np.isnan(std) else std)}"


def _stats(values: Sequence[float]) -> tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan")


def summary_rows(
    name: str,
    act: str,
    row: pd.Series,
    runs: pd.DataFrame,
    outputs: dict,
    labels: np.ndarray,
    refs: dict,
) -> dict[str, Any]:
    """Una fila de la tabla final para la config `row` (todas sus semillas)."""
    sel = runs[runs["hash"] == row["hash"]]
    out_frac = [out_of_range_fraction(outputs[r]["y_score"]) for r in sel["run"]]
    sat = (
        [saturated_fraction(outputs[r]["y_score"]) for r in sel["run"]]
        if act == "sigmoid"
        else [np.nan]
    )
    sec = [
        label_metrics(labels[outputs[r]["idx"]], outputs[r]["y_score"], SECONDARY_THRESHOLD)
        for r in sel["run"]
    ]
    mse_m, mse_s = row["mse_mean"], row["mse_std"]
    return {
        "experimento": name,
        "modelo": MODEL_NAMES[act],
        "optimizador": (
            f"{sel['training.optimizer.kind'].iloc[0]} η={row['training.optimizer.lr']:g}"
        ),
        "modo": mode_label(row["training.batch_size"]),
        "épocas": int(sel["epochs"].iloc[0]),
        "n": int(row["n_ok"]),
        "mse_mean": mse_m,
        "mse_std": mse_s,
        "mejora_vs_constante": 1 - mse_m / refs["constante"],
        "exceso_sobre_mc": mse_m - refs["minimos_cuadrados"],
        "fuera_0_1": _stats(out_frac),
        "saturada": _stats(sat) if act == "sigmoid" else (np.nan, np.nan),
        "s_por_epoca": float(sel["s_per_epoch"].mean()),
        **{f"{k}@0.5": _stats([s[k] for s in sec]) for k in ("accuracy", "f1")},
    }


def _display_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    def pct(t: tuple[float, float]) -> str:
        return "—" if np.isnan(t[0]) else _pm(100 * t[0], 100 * t[1], "{:.1f}") + " %"

    return pd.DataFrame(
        [
            {
                "experimento": r["experimento"],
                "modelo": r["modelo"],
                "entrenamiento": f"{r['optimizador']} · {r['modo']} · {r['épocas']} ép.",
                "MSE train": f"{r['mse_mean']:.4g} ± {np.nan_to_num(r['mse_std']):.1g}",
                "mejora vs const.": f"{100 * r['mejora_vs_constante']:.1f} %",
                "MSE − MC": f"{r['exceso_sobre_mc']:+.2e}",
                "fuera de [0,1]": pct(r["fuera_0_1"]),
                "neurona saturada": pct(r["saturada"]),
                "s/época": f"{r['s_por_epoca']:.3g}",
                "F1@0.5 (secund.)": _pm(*r["f1@0.5"], "{:.3f}"),
            }
            for r in rows
        ]
    )


def _load(results_dir: Path, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = load_runs(results_dir / name)
    return runs, load_histories(results_dir / name)


def main(argv: Sequence[str] | None = None) -> int:
    """Genera las figuras, tabla.csv y resumen.json. Devuelve 0, o 2 si faltan corridas."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        prog="python -m analysis.ej1_learning",
        description="Figuras del estudio de aprendizaje del Ej1 (F10) desde results/.",
    )
    parser.add_argument("--results-dir", default="results", help="carpeta raíz de resultados")
    parser.add_argument("--out-dir", default="figures/ej1/learning", help="carpeta de las figuras")
    args = parser.parse_args(argv)
    results_dir, out_dir = Path(args.results_dir), Path(args.out_dir)

    missing = [n for n in EXPERIMENTS if not list((results_dir / n).glob("*/metrics.json"))]
    if missing:
        print(
            f"Faltan corridas de: {', '.join(missing)} (correr experiments.runner)", file=sys.stderr
        )
        return 2

    learning, h_learning = _load(results_dir, "ej1_learning")
    long_runs, h_long = _load(results_dir, "ej1_long")
    beta_runs, _ = _load(results_dir, "ej1_beta")
    scaling_runs, _ = _load(results_dir, "ej1_scaling")

    X, y, labels = dataset_arrays(results_dir / "ej1_learning" / learning["run"].iloc[0])
    refs = {"constante": constant_mse(y), "minimos_cuadrados": least_squares_mse(X, y)}

    keys = ["model.output_activation", "training.optimizer.lr", "training.batch_size"]
    t_learning = per_config(learning, keys)
    t_learning["modo"] = t_learning["training.batch_size"].map(mode_label)
    best = best_by_model(t_learning)
    t_long = per_config(long_runs, keys)
    long_best = best_by_model(t_long)
    plateaus = {}
    for act in best:
        plateaus[act] = {
            "ej1_learning": plateau(curve(h_learning, best[act]["hash"])["mean"].to_numpy()),
            "ej1_long": plateau(curve(h_long, long_best[act]["hash"])["mean"].to_numpy()),
        }

    outputs = run_outputs(results_dir / "ej1_learning", learning)
    outputs_long = run_outputs(results_dir / "ej1_long", long_runs)
    outputs_beta = run_outputs(results_dir / "ej1_beta", beta_runs)

    t_beta = per_config(beta_runs, ["model.beta", "dataset.normalize"])
    sat = {
        h: [saturated_fraction(outputs_beta[r]["y_score"]) for r in g["run"]]
        for h, g in beta_runs.groupby("hash")
    }
    t_beta["sat_mean"] = t_beta["hash"].map(lambda h: np.mean(sat[h]))
    t_beta["sat_std"] = t_beta["hash"].map(
        lambda h: np.std(sat[h], ddof=1) if len(sat[h]) > 1 else np.nan
    )

    # ej1_scaling usa el mismo optimizador para ambos modelos: una config por modelo × scaler.
    t_scaling = per_config(scaling_runs, [*keys, "dataset.normalize"])
    if t_scaling.duplicated(["model.output_activation", "dataset.normalize"]).any():
        raise ValueError("ej1_scaling: más de una config por modelo y normalización")

    rows = [
        summary_rows("ej1_learning", act, best[act], learning, outputs, labels, refs)
        for act in best
    ]
    rows += [
        summary_rows("ej1_long", act, long_best[act], long_runs, outputs_long, labels, refs)
        for act in long_best
    ]
    display = _display_table(rows)

    lin = learning[learning["hash"] == best["identity"]["hash"]]
    lin_scores = np.concatenate([outputs[r]["y_score"] for r in lin["run"]])
    lin_frac = _stats([out_of_range_fraction(outputs[r]["y_score"]) for r in lin["run"]])

    apply_style()
    figures = {
        "E1-A-a_train_mse": fig_train_mse(h_learning, best, refs),
        "E1-A_lr_sensitivity": fig_lr_sensitivity(t_learning, refs),
        "E1-A-b_plateau": fig_plateau(
            h_learning,
            h_long,
            best,
            long_best,
            refs,
            {a: p["ej1_long"] for a, p in plateaus.items()},
        ),
        "E1-A-b_linear_outputs": fig_linear_outputs(
            lin_scores,
            *lin_frac,
            f"Lineal (η = {best['identity']['training.optimizer.lr']:g}, "
            f"{mode_label(best['identity']['training.batch_size'])})",
        ),
        "E1-A-b_saturation_beta": fig_saturation_beta(t_beta),
        "E1-A-b_scaling": fig_scaling(t_scaling, refs),
        "E1-A-c_summary_table": fig_summary_table(display),
    }
    assert tuple(figures) == FIGURES
    for name, fig in figures.items():
        save_figure(fig, out_dir / name)

    out_dir.mkdir(parents=True, exist_ok=True)
    display.to_csv(out_dir / "tabla.csv", index=False)

    def records(t: pd.DataFrame) -> list[dict[str, Any]]:
        return json.loads(t.to_json(orient="records"))

    resumen = {
        "referencias": refs,
        "mejor": {MODEL_NAMES[a]: records(r.to_frame().T)[0] for a, r in best.items()},
        "mejor_long": {MODEL_NAMES[a]: records(r.to_frame().T)[0] for a, r in long_best.items()},
        "meseta": {MODEL_NAMES[a]: p for a, p in plateaus.items()},
        "tabla": json.loads(json.dumps(rows, default=float)),
        "error_por_clase": {
            MODEL_NAMES[act]: [
                mse_by_class(labels[outputs[r]["idx"]], y[outputs[r]["idx"]], outputs[r]["y_score"])
                for r in learning.loc[learning["hash"] == row["hash"], "run"]
            ]
            for act, row in best.items()
        },
        "learning": records(t_learning),
        "beta": records(t_beta),
        "scaling": records(t_scaling),
    }
    (out_dir / "resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False, default=float) + "\n", encoding="utf-8"
    )
    print(display.to_string(index=False))
    print(
        f"Referencias: constante {refs['constante']:.5g} · "
        f"mínimos cuadrados {refs['minimos_cuadrados']:.5g}"
    )
    print(f"Figuras en {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
