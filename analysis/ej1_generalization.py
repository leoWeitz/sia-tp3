"""Ej1 · generalización, mejor modelo y umbral recomendado (F11): figuras 1–7, tablas y resumen.

    python -m analysis.ej1_generalization [--results-dir results]
                                          [--out-dir figures/ej1/generalization]
                                          [--criterion CRITERIO]

Solo lee results/ (CLAUDE.md §2.6). Antes hay que correr:

    python -m experiments.runner experiments/configs/ej1/cv.json
    python -m experiments.runner experiments/configs/ej1/split_strategy.json
    python -m experiments.ej1_best_fold experiments/configs/ej1/best_fold.json
    python -m experiments.runner experiments/configs/ej1/final.json --final-eval   # una vez

Solo ej1_cv es obligatorio. Sin ej1_split_strategy o ej1_best_fold se omiten
sus figuras; sin final_eval.json, las del test (FINAL_FIGURES).

El modelo aprende la probabilidad de BigModel (MSE); las métricas binarias se
calculan contra `flagged_fraud`, que se lee del CSV por índice de fila
(predictions.npz › idx y el test_idx.npy del TEST) y nunca entra al entrenamiento.

Protocolo (docs/fases/F11):
- Selección: la config de ej1_cv con menor val_loss media, entre las que no
  divergieron en ninguna corrida.
- Umbral (04-matematica §7): threshold_sweep sobre las predicciones
  out-of-fold (OOF) de esa config, concatenando todas las semillas. Se
  reportan todos los criterios (CRITERIA) y se recomienda uno (--criterion,
  RECOMMENDED por defecto). El umbral se fija acá y se aplica tal cual al TEST.
- Costos: c_FP = 1 y c_FN = razón (COST_RATIOS).

Figuras (FIGURES siempre, FINAL_FIGURES con final_eval.json) en <out-dir>
como .png y .pdf, más cv.csv, criterios.csv, costos.csv y resumen.json.
"""

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.common import PALETTE, apply_style, load_runs, save_figure
from core.serialization import load_checkpoint
from core.thresholds import (
    auc_trapezoid,
    average_precision,
    pr_curve,
    roc_curve,
    select_threshold,
    threshold_sweep,
)

CV, SPLIT, BEST_FOLD, FINAL = "ej1_cv", "ej1_split_strategy", "ej1_best_fold", "ej1_final"
LABEL = "flagged_fraud"
# flagged_fraud = 1 ⇔ probabilidad de BigModel > 0.85 (docs/datos/fraud_dataset.md).
BIGMODEL_THRESHOLD = 0.85
# Franja de probabilidad de BigModel "cerca del corte" para ubicar los errores en TEST.
NEAR_CUT = (0.7, 0.95)
COST_RATIOS = (1, 2, 5, 10, 20)
COST_RATIO_TABLE = 5
MIN_RECALLS = (0.8, 0.9)
CRITERIA = (
    "f1",
    "f2",
    "youden",
    f"cost:{COST_RATIO_TABLE}",
    *(f"precision_at_recall:{r}" for r in MIN_RECALLS),
)
RECOMMENDED = "f2"
TINY_ROWS = 100_000
TINY_REPEATS = 7
FIGURES = (
    "E1-B-a_cv_table",
    "E1-B-b_split_strategy",
    "E1-B-b_train_fraction",
    "E1-B-b_best_fold",
    "E1-B-c_pr_roc",
    "E1-B-c_threshold_metrics",
    "E1-B-c_cost_threshold",
    "E1-B-c_tiny",
)
FINAL_FIGURES = ("E1-B-c_test_confusion",)
_SPLIT_FIGURES = ("E1-B-b_split_strategy", "E1-B-b_train_fraction")
_INK = "#0b0b0b"  # texto principal (el de analysis.common)
STRATEGY_ORDER = ("holdout aleatorio", "holdout estratificado", "5-fold estratificado")


# --- Datos ---


def _read_json(path: str | PathLike) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_labels(csv_path: str | PathLike) -> np.ndarray:
    """flagged_fraud (p,) del CSV, en el orden de las filas (el de idx)."""
    return pd.read_csv(csv_path, usecols=[LABEL])[LABEL].to_numpy().astype(int)


def _load_predictions(run_dir: Path, name: str = "predictions.npz") -> dict[str, np.ndarray]:
    z = np.load(run_dir / name)
    return {
        "idx": z["idx"],
        "y_true": z["y_true"].ravel(),
        "y_score": z["y_score"].ravel(),
    }


def select_config(runs: pd.DataFrame) -> str:
    """Hash de la config con menor val.loss media.

    Solo compite una config si terminó ok en todas sus corridas (semillas × folds).
    """
    ok = runs.groupby("hash")["status"].apply(lambda s: bool((s == "ok").all()))
    candidates = runs[runs["hash"].isin(ok[ok].index)]
    if candidates.empty:
        raise ValueError("Ninguna config de ej1_cv terminó ok en todas sus corridas")
    return str(candidates.groupby("hash")["val.loss"].mean().idxmin())


def oof_predictions(results_dir: Path, runs: pd.DataFrame, hash_: str) -> pd.DataFrame:
    """Predicciones out-of-fold de una config: columnas seed, fold, idx, y_true, y_score.

    Verifica que cada semilla cubra cada fila de DESARROLLO exactamente una vez
    y que todas las semillas cubran las mismas filas (ValueError si no).
    """
    frames = []
    for _, run in runs[runs["hash"] == hash_].iterrows():
        p = _load_predictions(Path(results_dir) / run["run"])
        frames.append(pd.DataFrame({"seed": run["seed"], "fold": run["fold"], **p}))
    if not frames:
        raise ValueError(f"No hay corridas de la config {hash_}")
    df = pd.concat(frames, ignore_index=True)
    reference = None
    for seed, g in df.groupby("seed"):
        idx = np.sort(g["idx"].to_numpy())
        if np.unique(idx).size != idx.size:
            raise ValueError(f"OOF de la semilla {seed}: hay filas en más de un fold")
        if reference is None:
            reference = idx
        elif not np.array_equal(idx, reference):
            raise ValueError(f"OOF de la semilla {seed}: no cubre las mismas filas que las demás")
    return df


def _safe_ap(y: np.ndarray, s: np.ndarray) -> float:
    return average_precision(y, s) if 0 < y.sum() else float("nan")


def _roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    return auc_trapezoid(fpr, tpr)


# --- Umbral ---


def choose_threshold(y: np.ndarray, s: np.ndarray, criterion: str) -> dict[str, Any]:
    """Umbral elegido por un criterio de CRITERIA: 'f1', 'f2', 'youden', 'cost:<c_FN/c_FP>'
    o 'precision_at_recall:<recall mínimo>' (04-matematica §7). Devuelve la fila de
    select_threshold más la clave del criterio.
    """
    name, _, param = criterion.partition(":")
    if name in ("f1", "f2", "youden") and not param:
        result = select_threshold(threshold_sweep(y, s), name)
    elif name == "cost" and param:
        result = select_threshold(threshold_sweep(y, s, cost_fn=float(param), cost_fp=1.0), "cost")
    elif name == "precision_at_recall" and param:
        result = select_threshold(threshold_sweep(y, s), name, min_recall=float(param))
    else:
        raise ValueError(f"Criterio desconocido: {criterion!r}. Opciones: {list(CRITERIA)}")
    return {**result, "criterion": criterion}


def per_1000(y: np.ndarray, s: np.ndarray, threshold: float) -> dict[str, float]:
    """Qué pasa por cada 1000 transacciones al marcar fraude si score ≥ threshold."""
    y = np.asarray(y).astype(bool)
    flagged = np.asarray(s) >= threshold
    scale = 1000.0 / len(y)
    return {
        "fraudes": scale * y.sum(),
        "detectados": scale * (y & flagged).sum(),
        "no_detectados": scale * (y & ~flagged).sum(),
        "falsas_alarmas": scale * (~y & flagged).sum(),
        "alertas": scale * flagged.sum(),
    }


def _row(y: np.ndarray, s: np.ndarray, chosen: Mapping[str, Any]) -> dict[str, Any]:
    t = chosen["threshold"]
    return {
        "criterio": chosen["criterion"],
        "threshold": t,
        "precision": chosen["precision"],
        "recall": chosen["recall"],
        "f1": chosen["f1"],
        "f2": chosen["f2"],
        "fpr": chosen["fpr"],
        **{f"{k}_por_1000": v for k, v in per_1000(y, s, t).items()},
    }


def criteria_table(
    y: np.ndarray, s: np.ndarray, cost_ratio: float = COST_RATIO_TABLE
) -> pd.DataFrame:
    """Una fila por criterio de CRITERIA: umbral, métricas y cuentas por 1000 transacciones.

    El criterio de costo usa c_FN / c_FP = cost_ratio.
    """
    criteria = [c if not c.startswith("cost:") else f"cost:{cost_ratio:g}" for c in CRITERIA]
    return pd.DataFrame([_row(y, s, choose_threshold(y, s, c)) for c in criteria])


def cost_sensitivity(
    y: np.ndarray, s: np.ndarray, ratios: Sequence[float] = COST_RATIOS
) -> pd.DataFrame:
    """Umbral de costo mínimo para cada razón c_FN/c_FP (c_FP = 1) y lo que implica."""
    rows = []
    for ratio in ratios:
        row = _row(y, s, choose_threshold(y, s, f"cost:{ratio:g}"))
        rows.append({"ratio": ratio, **{k: v for k, v in row.items() if k != "criterio"}})
    return pd.DataFrame(rows)


def threshold_stability(oof: pd.DataFrame, labels: np.ndarray, criterion: str) -> pd.DataFrame:
    """Umbral del criterio elegido en la validación de cada (semilla, fold): ¿es estable?"""
    rows = []
    for (seed, fold), g in oof.groupby(["seed", "fold"]):
        y = labels[g["idx"].to_numpy()]
        try:
            t = choose_threshold(y, g["y_score"].to_numpy(), criterion)["threshold"]
        except ValueError:  # p. ej. un fold sin positivos o que no alcanza el recall mínimo
            t = float("nan")
        rows.append({"seed": seed, "fold": fold, "threshold": t})
    return pd.DataFrame(rows)


# --- Tablas por experimento ---


def cv_table(results_dir: Path, runs: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Una fila por config de ej1_cv: η, β, val MSE (folds × semillas) y AP / ROC-AUC OOF.

    AP y ROC-AUC se calculan sobre el OOF de cada semilla y se promedian entre semillas.
    """
    rows = []
    for hash_, g in runs.groupby("hash"):
        ok = bool((g["status"] == "ok").all())
        ap, auc = [], []
        if ok:
            oof = oof_predictions(results_dir, runs, hash_)
            for _, o in oof.groupby("seed"):
                y, s = labels[o["idx"].to_numpy()], o["y_score"].to_numpy()
                ap.append(_safe_ap(y, s))
                auc.append(_roc_auc(y, s))
        rows.append(
            {
                "hash": hash_,
                "lr": g["training.optimizer.lr"].iloc[0],
                "beta": g["model.beta"].iloc[0],
                "n": len(g),
                "n_ok": int((g["status"] == "ok").sum()),
                "val_mse_mean": g["val.mse"].mean(),
                "val_mse_std": g["val.mse"].std(ddof=1),
                "train_mse_mean": g["train.mse"].mean(),
                "ap_mean": np.mean(ap) if ap else np.nan,
                "ap_std": np.std(ap, ddof=1) if len(ap) > 1 else np.nan,
                "roc_auc_mean": np.mean(auc) if auc else np.nan,
                "best_epoch_median": g["best_epoch"].median(),
            }
        )
    return pd.DataFrame(rows).sort_values(["lr", "beta"]).reset_index(drop=True)


def strategy_name(split: Mapping[str, Any]) -> str:
    """Nombre de la estrategia de partición para las figuras."""
    kind = split["kind"]
    if kind == "kfold":
        return f"{split['k']}-fold" + (" estratificado" if split["stratified"] else "")
    if kind == "holdout":
        return "holdout " + ("estratificado" if split["stratified"] else "aleatorio")
    return kind


def _split_of(run: pd.Series) -> dict[str, Any]:
    kind = run["dataset.split.kind"]
    split = {"kind": kind, "stratified": bool(run["dataset.split.stratified"])}
    if kind == "kfold":
        split["k"] = int(run["dataset.split.k"])
    else:
        split["ratio"] = float(run["dataset.split.ratio"])
    return split


def split_estimates(
    results_dir: Path, runs: pd.DataFrame, labels: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(particiones, estimaciones) de ej1_split_strategy.

    particiones: una fila por corrida (holdout o fold) con la tasa de fraude de
    su validación, AP y MSE. estimaciones: la estimación de generalización de
    cada semilla, que en k-fold es la media de sus folds.
    """
    rows = []
    for _, run in runs.iterrows():
        split = _split_of(run)
        p = _load_predictions(Path(results_dir) / run["run"])
        y = labels[p["idx"]]
        rows.append(
            {
                "strategy": strategy_name(split),
                "ratio": split.get("ratio", np.nan),
                "seed": run["seed"],
                "fold": run["fold"],
                "fraud_rate": float(y.mean()),
                "ap": _safe_ap(y, p["y_score"]),
                "mse": run["val.mse"],
                "train_mse": run["train.mse"],
            }
        )
    partitions = pd.DataFrame(rows)
    estimates = (
        partitions.groupby(["strategy", "ratio", "seed"], dropna=False)[["ap", "mse", "train_mse"]]
        .mean()
        .reset_index()
    )
    return partitions, estimates


def best_fold_table(results_dir: Path, labels: np.ndarray) -> pd.DataFrame:
    """Una fila por partición de ej1_best_fold: mejor fold (val y pseudo-test) vs todo el resto."""
    rows = []
    for path in sorted(Path(results_dir).glob("partition_s*/partition.json")):
        part = _read_json(path)
        best = next(f for f in part["folds"] if f["fold"] == part["best_fold"])
        val = _load_predictions(Path(results_dir) / best["run"])
        ps_best = _load_predictions(path.parent, f"pseudo_f{best['fold']}.npz")
        ps_rest = _load_predictions(path.parent, "pseudo_rest.npz")
        y_ps = labels[ps_best["idx"]]
        rows.append(
            {
                "seed": part["seed"],
                "best_fold": part["best_fold"],
                "rest_epochs": part["rest"]["epochs"],
                "best_val_mse": best["val_loss"],
                "best_pseudo_mse": best["pseudo_test"]["mse"],
                "rest_pseudo_mse": part["rest"]["pseudo_test"]["mse"],
                "folds_pseudo_mse_mean": float(
                    np.mean([f["pseudo_test"]["mse"] for f in part["folds"]])
                ),
                "best_val_ap": _safe_ap(labels[val["idx"]], val["y_score"]),
                "best_pseudo_ap": _safe_ap(y_ps, ps_best["y_score"]),
                "rest_pseudo_ap": _safe_ap(y_ps, ps_rest["y_score"]),
            }
        )
    return pd.DataFrame(rows)


def _binary_metrics(y: np.ndarray, s: np.ndarray, threshold: float) -> dict[str, Any]:
    row = threshold_sweep(y, s, thresholds=np.array([threshold])).iloc[0]
    tp, fp, tn, fn = (int(row[k]) for k in ("TP", "FP", "TN", "FN"))
    return {
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1"]),
        "f2": float(row["f2"]),
        "fpr": float(row["fpr"]),
        "accuracy": (tp + tn) / len(y),
        "average_precision": _safe_ap(y, s),
        "roc_auc": _roc_auc(y, s),
        # Filas = real, columnas = predicho, positivo (fraude) primero (04-matematica §6).
        "confusion": [[tp, fn], [fp, tn]],
    }


def near_cut_fraction(prob: np.ndarray, y: np.ndarray, flagged: np.ndarray) -> float:
    """Fracción de los errores (FN + FP) con probabilidad de BigModel dentro de NEAR_CUT.

    prob (n,): probabilidad de BigModel. y, flagged (n,): etiqueta real y predicha. NaN sin errores.
    """
    errors = np.asarray(y).astype(bool) != np.asarray(flagged).astype(bool)
    if not errors.any():
        return float("nan")
    lo, hi = NEAR_CUT
    return float(np.mean((prob[errors] >= lo) & (prob[errors] <= hi)))


def final_results(final_dir: Path, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    """Métricas en TEST de cada modelo de ej1_final con el umbral elegido en OOF."""
    report = _read_json(final_dir / "final_eval.json")
    test_idx = np.sort(np.load(report["test"]["indices_path"]))
    y = labels[test_idx]
    models = []
    for model in report["models"]:
        out = Path(model["model"]).parent
        if not (out / "test_predictions.npz").exists():
            out = final_dir / f"final_{model['label']}"
        z = np.load(out / "test_predictions.npz")
        s, prob = z["y_score"].ravel(), z["y_true"].ravel()
        models.append(
            {
                "label": model["label"],
                "mse": float(np.mean((s - prob) ** 2)),
                **_binary_metrics(y, s, threshold),
                "por_1000": per_1000(y, s, threshold),
                "errores_cerca_del_corte": near_cut_fraction(prob, y, s >= threshold),
            }
        )
    keys = [k for k, v in models[0].items() if isinstance(v, (int, float))]
    return {
        "threshold": threshold,
        "n": int(len(y)),
        "fraud_rate": float(y.mean()),
        "epochs": report["epochs"],
        "selected_from": report["selected_from"],
        "models": models,
        "mean": {k: float(np.mean([m[k] for m in models])) for k in keys},
        "std": {
            k: float(np.std([m[k] for m in models], ddof=1)) if len(models) > 1 else None
            for k in keys
        },
        "confusion_mean": np.mean([m["confusion"] for m in models], axis=0).tolist(),
        "confusion_std": (
            np.std([m["confusion"] for m in models], axis=0, ddof=1).tolist()
            if len(models) > 1
            else None
        ),
        "por_1000_mean": {
            k: float(np.mean([m["por_1000"][k] for m in models])) for k in models[0]["por_1000"]
        },
        "fraccion_cerca_del_corte": float(np.mean((prob >= NEAR_CUT[0]) & (prob <= NEAR_CUT[1]))),
        # BigModel en el mismo TEST: su probabilidad es el target (y_true de las predicciones).
        "bigmodel": _binary_metrics(y, prob, BIGMODEL_THRESHOLD)
        | {"threshold": BIGMODEL_THRESHOLD},
    }


def tiny_stats(model_path: Path, rows: int = TINY_ROWS, repeats: int = TINY_REPEATS) -> dict:
    """Tamaño del TinyModel: parámetros, bytes del model.npz y tiempo de inferencia de `rows` filas.

    Solo inferencia (predict sobre datos ya normalizados, al azar): no entrena.
    Se reporta la mediana de `repeats` repeticiones.
    """
    network, _, _ = load_checkpoint(model_path)
    X = np.random.default_rng(0).standard_normal((rows, network.layers[0].W.shape[0]))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        network.predict(X)
        times.append(time.perf_counter() - start)
    return {
        "model": str(model_path),
        "n_params": int(network.n_params),
        "bytes": int(Path(model_path).stat().st_size),
        "rows": rows,
        "seconds_median": float(np.median(times)),
    }


# --- Figuras ---


def _table_figure(df: pd.DataFrame, title: str, highlight: int | None = None) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(min(18, 2.0 + 1.9 * len(df.columns)), 1.2 + 0.5 * len(df)))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=list(df.columns), loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.5)
    for (r, _), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
        elif highlight is not None and r - 1 == highlight:
            cell.set_facecolor("#e8f0fb")
    ax.set_title(title)
    return fig


def _pm(mean: float, std: float, fmt: str = "{:.4g}") -> str:
    if mean is None or np.isnan(mean):
        return "—"
    if std is None or np.isnan(std):
        return fmt.format(mean)
    return f"{fmt.format(mean)} ± {fmt.format(std)}"


def _pm_column(table: pd.DataFrame, name: str, fmt: str = "{:.4g}") -> list[str]:
    """Columna "media ± desvío" a partir de <name>_mean y <name>_std."""
    pairs = zip(table[f"{name}_mean"], table[f"{name}_std"], strict=True)
    return [_pm(m, s, fmt) for m, s in pairs]


def fig_cv_table(table: pd.DataFrame, selected: str) -> Any:
    display = pd.DataFrame(
        {
            "η": table["lr"].map("{:g}".format),
            "β": table["beta"].map("{:g}".format),
            "val MSE": _pm_column(table, "val_mse"),
            "AP (OOF)": _pm_column(table, "ap", "{:.3f}"),
            "ROC-AUC (OOF)": table["roc_auc_mean"].map("{:.3f}".format),
            "mejor época (mediana)": table["best_epoch_median"].map("{:g}".format),
            "ok": [f"{a}/{b}" for a, b in zip(table["n_ok"], table["n"], strict=True)],
        }
    )
    highlight = int(np.flatnonzero(table["hash"] == selected)[0])
    return _table_figure(
        display,
        "E1-B-a · 5-fold estratificado × semillas en DESARROLLO (resaltada: la elegida)",
        highlight,
    )


def fig_split_strategy(partitions: pd.DataFrame, estimates: pd.DataFrame) -> Any:
    import matplotlib.pyplot as plt

    main = estimates[(estimates["ratio"].isna()) | (estimates["ratio"] == 0.8)]
    parts = partitions[(partitions["ratio"].isna()) | (partitions["ratio"] == 0.8)]
    order = [s for s in STRATEGY_ORDER if s in set(main["strategy"])]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax1.boxplot(
        [main.loc[main["strategy"] == s, "mse"].dropna() * 1e3 for s in order], tick_labels=order
    )
    ax1.set_ylabel("MSE de validación (×10⁻³)\nuna estimación por semilla")
    ax1.set_title("Estimación de la generalización")
    ax2.boxplot(
        [parts.loc[parts["strategy"] == s, "fraud_rate"] * 100 for s in order], tick_labels=order
    )
    ax2.set_ylabel("% de fraude en la partición de validación")
    ax2.set_title("Representatividad de la validación")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", labelrotation=12)
    fig.suptitle("E1-B-b · Estrategia de partición (80/20 o 5-fold, dentro de DESARROLLO)")
    fig.tight_layout()
    return fig


def fig_train_fraction(estimates: pd.DataFrame) -> Any:
    import matplotlib.pyplot as plt

    g = (
        estimates[estimates["strategy"] == "holdout estratificado"]
        .groupby("ratio")[["train_mse", "mse", "ap"]]
        .agg(["mean", "std"])
    )
    x = g.index.to_numpy() * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for col, label, color in (
        ("train_mse", "train", PALETTE[0]),
        ("mse", "validación", PALETTE[1]),
    ):
        ax1.errorbar(
            x,
            g[(col, "mean")],
            yerr=g[(col, "std")],
            marker="o",
            capsize=4,
            color=color,
            label=label,
        )
    ax1.set_xlabel("% de DESARROLLO usado para entrenar")
    ax1.set_ylabel("MSE vs BigModel")
    ax1.legend()
    ax2.errorbar(
        x, g[("ap", "mean")], yerr=g[("ap", "std")], marker="o", capsize=4, color=PALETTE[2]
    )
    ax2.set_xlabel("% de DESARROLLO usado para entrenar")
    ax2.set_ylabel("AP en validación")
    fig.suptitle(
        "E1-B-b · Métricas vs fracción de entrenamiento (holdout estratificado, media ± desvío)"
    )
    fig.tight_layout()
    return fig


def fig_best_fold(table: pd.DataFrame) -> Any:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    bars = (
        ("mejor fold\n(su validación)", "best_val", PALETTE[3]),
        ("mejor fold\n(pseudo-test)", "best_pseudo", PALETTE[1]),
        ("todo el resto\n(pseudo-test)", "rest_pseudo", PALETTE[0]),
    )
    for ax, metric, ylabel in ((ax1, "mse", "MSE vs BigModel"), (ax2, "ap", "AP")):
        cols = [f"{key}_{metric}" for _, key, _ in bars]
        means, stds = table[cols].mean(), table[cols].std(ddof=1)
        ax.bar([b[0] for b in bars], means, yerr=stds, capsize=6, color=[b[2] for b in bars])
        for i, (m, s) in enumerate(zip(means, stds, strict=True)):
            ax.annotate(
                _pm(m, s, "{:.4f}"),
                (i, m),
                textcoords="offset points",
                xytext=(0, 14),
                ha="center",
                fontsize=11,
            )
        ax.set_ylabel(ylabel)
    fig.suptitle(
        f"E1-B-b · Elegir el fold por su puntaje vs entrenar con todo "
        f"({len(table)} particiones, media ± desvío)"
    )
    fig.tight_layout()
    return fig


def fig_pr_roc(y: np.ndarray, s: np.ndarray, chosen: Mapping[str, Any], title: str) -> Any:
    import matplotlib.pyplot as plt

    prec, rec, _ = pr_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    ax1.plot(rec, prec, color=PALETTE[0], label=f"TinyModel (AP = {average_precision(y, s):.3f})")
    ax1.axhline(y.mean(), color=PALETTE[7], linestyle=":", label=f"azar ({y.mean():.3f})")
    ax1.plot(1, 1, "*", color=PALETTE[5], markersize=14, label="BigModel (AP = 1)")
    ax1.plot(
        chosen["recall"],
        chosen["precision"],
        "o",
        color=PALETTE[1],
        markersize=10,
        label=f"umbral {chosen['threshold']:.2f} ({chosen['criterion']})",
    )
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision-Recall")
    ax1.legend(fontsize=11, loc="lower left")
    ax2.plot(fpr, tpr, color=PALETTE[0], label=f"TinyModel (AUC = {auc_trapezoid(fpr, tpr):.3f})")
    ax2.plot([0, 1], [0, 1], color=PALETTE[7], linestyle=":", label="azar")
    ax2.plot(
        chosen["fpr"],
        chosen["recall"],
        "o",
        color=PALETTE[1],
        markersize=10,
        label=f"umbral {chosen['threshold']:.2f}",
    )
    ax2.set_xlabel("FPR")
    ax2.set_ylabel("TPR (recall)")
    ax2.set_title("ROC")
    ax2.legend(fontsize=11, loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def fig_threshold_metrics(
    y: np.ndarray, s: np.ndarray, criteria: pd.DataFrame, recommended: str
) -> Any:
    import matplotlib.pyplot as plt

    sweep = threshold_sweep(y, s)
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, m in enumerate(("precision", "recall", "f1", "f2")):
        ax.plot(sweep["threshold"], sweep[m], color=PALETTE[i], label=m)
    for j, row in criteria.iterrows():
        rec = row["criterio"] == recommended
        ax.axvline(
            row["threshold"],
            color=_INK if rec else "#898781",
            linewidth=2 if rec else 1,
            linestyle="-" if rec else "--",
        )
        ax.annotate(
            row["criterio"] + (" ★" if rec else ""),
            (row["threshold"], 0.02 + 0.07 * j),
            rotation=0,
            fontsize=10,
            xytext=(3, 0),
            textcoords="offset points",
        )
    ax.set_xlabel("Umbral t (fraude si score ≥ t)")
    ax.set_ylabel("Métrica (OOF)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="center left")
    ax.set_title("E1-B-c · Métricas vs umbral (OOF), umbrales de cada criterio (★ recomendado)")
    return fig


def fig_cost_threshold(costs: pd.DataFrame) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(costs["ratio"], costs["threshold"], "o-", color=PALETTE[0])
    for _, r in costs.iterrows():
        ax.annotate(
            f"recall {r['recall']:.2f}\n{r['falsas_alarmas_por_1000']:.0f} falsas/1000",
            (r["ratio"], r["threshold"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=10,
        )
    ax.set_xscale("log")
    ax.set_xticks(costs["ratio"], [f"{r:g}" for r in costs["ratio"]])
    ax.set_xlabel("Razón de costos c_FN / c_FP")
    ax.set_ylabel("Umbral de costo mínimo")
    ax.set_title("E1-B-c · Umbral óptimo vs razón de costos (OOF)")
    return fig


def fig_test_confusion(test: Mapping[str, Any]) -> Any:
    import matplotlib.pyplot as plt

    cm = np.array(test["confusion_mean"])
    sd = None if test["confusion_std"] is None else np.array(test["confusion_std"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.3]})
    ax1.imshow(cm, cmap="Blues")
    labels = ["fraude", "no fraude"]
    for i in range(2):
        for j in range(2):
            text = f"{cm[i, j]:.1f}" + ("" if sd is None else f"\n± {sd[i, j]:.1f}")
            color = "white" if cm[i, j] > cm.max() / 2 else _INK
            ax1.text(j, i, text, ha="center", va="center", color=color, fontsize=14)
    ax1.set_xticks([0, 1], labels)
    ax1.set_yticks([0, 1], labels)
    ax1.set_xlabel("Predicho")
    ax1.set_ylabel("Real")
    ax1.grid(False)
    ax1.set_title(f"Matriz de confusión (media de {len(test['models'])} modelos)")
    mean, std = test["mean"], test["std"]
    rows = [
        ("precision", "precision"),
        ("recall", "recall"),
        ("F1", "f1"),
        ("F2", "f2"),
        ("FPR", "fpr"),
        ("accuracy", "accuracy"),
        ("AP", "average_precision"),
        ("ROC-AUC", "roc_auc"),
        ("MSE vs BigModel", "mse"),
    ]
    big = test["bigmodel"]
    cell = [
        [
            n,
            _pm(mean[k], std[k], "{:.2e}" if k == "mse" else "{:.3f}"),
            f"{big[k]:.3f}" if k in big else "—",
        ]
        for n, k in rows
    ]
    ax2.axis("off")
    t = ax2.table(
        cellText=cell,
        colLabels=["métrica", "TinyModel", "BigModel (0.85)"],
        loc="center",
        cellLoc="center",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(12)
    t.scale(1.0, 1.6)
    fig.suptitle(
        f"E1-B-c · TEST (n = {test['n']}), umbral {test['threshold']:.2f} elegido en validación"
    )
    fig.tight_layout()
    return fig


def fig_tiny(tiny: Mapping[str, Any], architecture: str) -> Any:
    rows = pd.DataFrame(
        [
            ("arquitectura", architecture),
            ("parámetros", f"{tiny['n_params']}"),
            ("model.npz", f"{tiny['bytes'] / 1024:.1f} KiB"),
            (
                f"inferencia de {tiny['rows']:,} transacciones",
                f"{tiny['seconds_median'] * 1000:.1f} ms",
            ),
        ],
        columns=["", "TinyModel"],
    )
    return _table_figure(rows, "E1-B-c · ¿Es tiny? (mediana de la inferencia en esta máquina)")


# --- Main ---


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records"))


def _describe(values: Sequence[float]) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return {
        "n": int(v.size),
        "mean": float(v.mean()) if v.size else None,
        "std": float(v.std(ddof=1)) if v.size > 1 else None,
        "min": float(v.min()) if v.size else None,
        "max": float(v.max()) if v.size else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Genera figuras, tablas y resumen.json. Devuelve 0, o 2 si falta ej1_cv."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        prog="python -m analysis.ej1_generalization",
        description="Figuras del estudio de generalización y umbral del Ej1 (F11) desde results/.",
    )
    parser.add_argument("--results-dir", default="results", help="carpeta raíz de resultados")
    parser.add_argument(
        "--out-dir", default="figures/ej1/generalization", help="carpeta de las figuras"
    )
    parser.add_argument(
        "--criterion", default=RECOMMENDED, help=f"criterio recomendado ({', '.join(CRITERIA)})"
    )
    args = parser.parse_args(argv)
    results_dir, out_dir = Path(args.results_dir), Path(args.out_dir)
    if args.criterion not in CRITERIA:
        parser.error(f"--criterion tiene que ser uno de {list(CRITERIA)}")

    cv_dir = results_dir / CV
    if not list(cv_dir.glob("*/metrics.json")):
        print(f"Faltan corridas de {CV} (correr experiments.runner)", file=sys.stderr)
        return 2

    cv_runs = load_runs(cv_dir)
    first = _read_json(cv_dir / cv_runs["run"].iloc[0] / "config.json")
    labels = load_labels(first["dataset"]["path"])
    test_idx = np.load(first["dataset"]["holdout_test"]["indices_path"])

    selected = select_config(cv_runs)
    table = cv_table(cv_dir, cv_runs, labels)
    oof = oof_predictions(cv_dir, cv_runs, selected)
    y_oof, s_oof = labels[oof["idx"].to_numpy()], oof["y_score"].to_numpy()
    criteria = criteria_table(y_oof, s_oof)
    chosen = choose_threshold(y_oof, s_oof, args.criterion)
    costs = cost_sensitivity(y_oof, s_oof)
    stability = threshold_stability(oof, labels, args.criterion)
    sel_row = table[table["hash"] == selected].iloc[0]
    sel_run = cv_runs[cv_runs["hash"] == selected].iloc[0]

    dev_idx = np.setdiff1d(np.arange(len(labels)), test_idx)
    fold_rates = [
        float(labels[g["idx"].to_numpy()].mean()) for _, g in oof.groupby(["seed", "fold"])
    ]
    resumen: dict[str, Any] = {
        "tasa_fraude": {
            "total": float(labels.mean()),
            "test": float(labels[test_idx].mean()),
            "desarrollo": float(labels[dev_idx].mean()),
            "folds_cv": _describe(fold_rates),
            "n_test": int(len(test_idx)),
            "n_desarrollo": int(len(dev_idx)),
        },
        "seleccion": {"hash": selected, **_records(sel_row.to_frame().T)[0]},
        "cv": _records(table),
        "criterios": {r["criterio"]: r for r in _records(criteria)},
        "recomendacion": {
            "criterio": args.criterion,
            "threshold": chosen["threshold"],
            "oof": _row(y_oof, s_oof, chosen),
        },
        "costos": _records(costs),
        "estabilidad": {
            "criterio": args.criterion,
            "por_fold": _records(stability),
            **_describe(stability["threshold"]),
        },
        "oof": {
            "n": int(len(y_oof)),
            "average_precision": average_precision(y_oof, s_oof),
            "roc_auc": _roc_auc(y_oof, s_oof),
        },
        "split_strategy": None,
        "best_fold": None,
        "test": None,
        "tiny": None,
    }

    apply_style()
    model_desc = (
        f"{sel_run['model.output_activation']} β={sel_row['beta']:g}, Adam η={sel_row['lr']:g}"
    )
    figures: dict[str, Any] = {
        "E1-B-a_cv_table": fig_cv_table(table, selected),
        "E1-B-c_pr_roc": fig_pr_roc(
            y_oof, s_oof, chosen, f"E1-B-c · Curvas OOF de la config elegida ({model_desc})"
        ),
        "E1-B-c_threshold_metrics": fig_threshold_metrics(y_oof, s_oof, criteria, args.criterion),
        "E1-B-c_cost_threshold": fig_cost_threshold(costs),
    }

    split_dir = results_dir / SPLIT
    if list(split_dir.glob("*/metrics.json")):
        partitions, estimates = split_estimates(split_dir, load_runs(split_dir), labels)
        figures["E1-B-b_split_strategy"] = fig_split_strategy(partitions, estimates)
        figures["E1-B-b_train_fraction"] = fig_train_fraction(estimates)
        by = ["strategy", "ratio"]
        resumen["split_strategy"] = {
            "estimaciones": _records(
                estimates.groupby(by, dropna=False)[["ap", "mse", "train_mse"]]
                .agg(["mean", "std", "count"])
                .pipe(lambda d: d.set_axis([f"{a}_{b}" for a, b in d.columns], axis=1))
                .reset_index()
            ),
            "tasa_fraude_validacion": _records(
                partitions.groupby(by, dropna=False)["fraud_rate"]
                .agg(["mean", "std", "min", "max", "count"])
                .reset_index()
            ),
        }
    else:
        print(f"Sin corridas de {SPLIT}: se omiten {', '.join(_SPLIT_FIGURES)}", file=sys.stderr)

    bf_dir = results_dir / BEST_FOLD
    if list(bf_dir.glob("partition_s*/partition.json")):
        bf = best_fold_table(bf_dir, labels)
        figures["E1-B-b_best_fold"] = fig_best_fold(bf)
        resumen["best_fold"] = {
            "particiones": _records(bf),
            **{c: _describe(bf[c]) for c in bf.columns if c not in ("seed", "best_fold")},
        }
    else:
        print(f"Sin particiones de {BEST_FOLD}: se omite E1-B-b_best_fold", file=sys.stderr)

    final_dir = results_dir / FINAL
    tiny_model = cv_dir / sel_run["run"] / "model.npz"
    if (final_dir / "final_eval.json").exists():
        test = final_results(final_dir, labels, chosen["threshold"])
        resumen["test"] = test
        figures["E1-B-c_test_confusion"] = fig_test_confusion(test)
        tiny_model = Path(_read_json(final_dir / "final_eval.json")["models"][0]["model"])
    if tiny_model.exists():
        resumen["tiny"] = tiny_stats(tiny_model)
        layers = [len(first["dataset"]["features"]), 1]
        figures["E1-B-c_tiny"] = fig_tiny(resumen["tiny"], f"{layers}, {model_desc}")

    for name, fig in figures.items():
        save_figure(fig, out_dir / name)

    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "cv.csv", index=False)
    criteria.to_csv(out_dir / "criterios.csv", index=False)
    costs.to_csv(out_dir / "costos.csv", index=False)
    (out_dir / "resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False, default=float) + "\n", encoding="utf-8"
    )
    print(criteria.to_string(index=False))
    print(f"Recomendado ({args.criterion}): umbral {chosen['threshold']:.2f}")
    print(f"Figuras en {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
