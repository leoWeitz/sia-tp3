"""Ej3 · Grilla activación × arquitectura × η sobre la unión de dígitos (shift ±1 px).

Lee results/ej3v2_grid (tanh/ReLU × 4 redes × η de Adam ∈ {3e-4, 1e-3}, RandomShift ±1 px,
salida sigmoide + MSE, sobre datasets/digits_union.csv), results/ej3_search (la búsqueda
anterior, para el efecto del augmentation) y, si existen, results/ej3v2_final y
results/ej2_final_grid. Solo lee results/ (CLAUDE.md §2.6). Figuras en figures/ej3/grid/:

- E3-G1_ranking: las 10 mejores configuraciones (accuracy de validación).
- E3-G2_factor_effect: cuánto mueve la accuracy cada factor (mejor config por nivel).
- E3-G3_activation_lr: accuracy y épocas vs η por activación, con la red de la mejor.
- E3-G4_arch_activation: accuracy por red y activación, con el η de la mejor.
- E3-G5_augmentation: train vs validación y recall de 5 y 8 por augmentation (ej3_search).
- E3-G6_test_confusion: matriz de confusión media en test (ej3v2_final).
- E3-G7_per_class_test: recall por dígito en test, Ej2 final vs Ej3 final.
- E3-G8_hard_examples: errores en test del mejor modelo.

Uso:
    python -m analysis.ej3_grid [--results-dir results] [--out-dir figures/ej3/grid]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.common import PALETTE, apply_style, save_figure
from analysis.ej2_grid import AUG_LABELS, load_grid_runs, summarize
from data.loaders import load_digits_csv

GRID = "ej3v2_grid"
SEARCH = "ej3_search"
FINAL = "ej3v2_final"
EJ2_FINAL = "ej2_final_grid"
ACTIVATIONS = ("tanh", "relu")
ACTIVATION_LABELS = {"tanh": "tanh (Xavier)", "relu": "ReLU (He)"}
ACTIVATION_COLORS = {"tanh": PALETTE[0], "relu": PALETTE[1]}
# η y shift de la grilla (experiments/configs/ej3/grid.json). results/ej3v2_grid también
# tiene corridas con η = 1e-4, sin shift y con ±2 px de antes de recortar la grilla por
# plazo: no forman configuraciones completas y se descartan.
LRS = (3e-4, 1e-3)
AUG = 1
# Config de ej3_search con las tres variantes de augmentation para E3-G5.
SEARCH_CONFIG = {"activation": "relu", "arch": "[256, 128]", "lr": 1e-3}
ARCH_ORDER = ("[128]", "[128, 64, 32]", "[256]", "[256, 128]")
RARE_DIGITS = (5, 8)  # las clases escasas de la unión (785 y 585 muestras)
_MUTED = "#898781"


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def config_label(row: pd.Series) -> str:
    act = ACTIVATION_LABELS[row["activation"]].split(" (")[0]
    return f"{act} · {row['arch']} · η = {row['lr']:g} · {AUG_LABELS[row['aug']]}"


def factor_effects(grid: pd.DataFrame) -> pd.DataFrame:
    """Para cada factor, la mejor accuracy de cada nivel (optimizando el resto) y su rango."""
    rows = []
    for factor, label, fmt in (
        ("arch", "Arquitectura", str),
        ("lr", "η de Adam", lambda v: f"η = {v:g}"),
        ("activation", "Activación oculta", lambda v: ACTIVATION_LABELS[v].split(" (")[0]),
    ):
        best = grid.groupby(factor)["val_acc_mean"].max()
        rows.append(
            {
                "factor": label,
                "best_level": fmt(best.idxmax()),
                "worst_level": fmt(best.idxmin()),
                "best": best.max(),
                "worst": best.min(),
                "range": best.max() - best.min(),
            }
        )
    return pd.DataFrame(rows).sort_values("range", ascending=False, ignore_index=True)


def plot_ranking(grid: pd.DataFrame, out_dir: Path, top: int = 10) -> list[Path]:
    best = grid.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(best))
    colors = [ACTIVATION_COLORS[a] for a in best["activation"]]
    ax.errorbar(
        best["val_acc_mean"], y, xerr=best["val_acc_std"], fmt="none", ecolor=_MUTED, capsize=4
    )
    ax.scatter(best["val_acc_mean"], y, c=colors, s=80, zorder=3, edgecolors="white", linewidths=2)
    for yi, (_, r) in zip(y, best.iterrows(), strict=True):
        ax.annotate(
            f"{r['val_acc_mean']:.2f} ± {r['val_acc_std']:.2f}  ({int(r['n_seeds'])} sem.)",
            (r["val_acc_mean"] + r["val_acc_std"], yi),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=11,
        )
    ax.axvline(98.0, color=_MUTED, lw=1, ls="--")
    ax.set_yticks(y, [config_label(r) for _, r in best.iterrows()], fontsize=12)
    ax.set_xlabel("Accuracy de validación (%)")
    ax.set_title(f"Las {top} mejores de {len(grid)} configuraciones · Adam · unión")
    ax.grid(axis="y", visible=False)
    lo = (best["val_acc_mean"] - best["val_acc_std"]).min()
    hi = (best["val_acc_mean"] + best["val_acc_std"]).max()
    ax.set_xlim(min(lo, 98.0) - 0.1, hi + 0.5)
    fig.tight_layout()
    return save_figure(fig, out_dir / "E3-G1_ranking")


def plot_factor_effect(effects: pd.DataFrame, out_dir: Path) -> list[Path]:
    eff = effects.iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    y = np.arange(len(eff))
    ax.barh(y, eff["range"], color=PALETTE[0], height=0.55)
    for yi, (_, r) in zip(y, eff.iterrows(), strict=True):
        ax.annotate(
            f"{r['range']:.2f} puntos  ({r['worst_level']} {r['worst']:.2f}"
            f" → {r['best_level']} {r['best']:.2f})",
            (r["range"], yi),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=11,
        )
    ax.set_yticks(y, eff["factor"])
    ax.set_xlabel("Diferencia entre el mejor y el peor nivel (puntos de accuracy de validación)")
    ax.set_title("Cuánto mueve la accuracy cada factor (el resto, en su mejor valor)")
    ax.set_xlim(0, eff["range"].max() * 2.4)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save_figure(fig, out_dir / "E3-G2_factor_effect")


def plot_activation_lr(grid: pd.DataFrame, arch: str, aug: int, out_dir: Path) -> list[Path]:
    sub = grid[(grid["arch"] == arch) & (grid["aug"] == aug)]
    fig, (ax_acc, ax_ep) = plt.subplots(1, 2, figsize=(13, 5))
    for act in ACTIVATIONS:
        d = sub[sub["activation"] == act].sort_values("lr")
        kw = {"color": ACTIVATION_COLORS[act], "marker": "o", "capsize": 3}
        ax_acc.errorbar(d["lr"], d["val_acc_mean"], yerr=d["val_acc_std"], **kw)
        ax_ep.errorbar(d["lr"], d["epochs_mean"], yerr=d["epochs_std"], **kw)
    for ax in (ax_acc, ax_ep):
        ax.set_xscale("log")
        ax.set_xticks(list(LRS), [f"{lr:g}".replace(".", ",") for lr in LRS])
        ax.minorticks_off()
        ax.set_xlabel("Tasa de aprendizaje η de Adam (escala log)")
    ax_acc.set_ylabel("Accuracy de validación (%)")
    ax_acc.set_title("Accuracy final")
    ax_ep.set_ylabel("Épocas hasta early stopping")
    ax_ep.set_title("Velocidad")
    handles = [
        plt.Line2D([], [], color=ACTIVATION_COLORS[a], marker="o", label=ACTIVATION_LABELS[a])
        for a in ACTIVATIONS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Activación y η · red {arch} · {AUG_LABELS[aug]} · 3 semillas", fontsize=15)
    fig.tight_layout()
    return save_figure(fig, out_dir / "E3-G3_activation_lr")


def plot_arch_activation(grid: pd.DataFrame, lr: float, out_dir: Path) -> list[Path]:
    sub = grid[np.isclose(grid["lr"], lr)]
    archs = [a for a in ARCH_ORDER if a in set(sub["arch"])]
    params = sub.groupby("arch")["n_params"].first()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(archs))
    for act in ACTIVATIONS:
        d = sub[sub["activation"] == act].set_index("arch").loc[archs]
        ax.errorbar(
            x,
            d["val_acc_mean"],
            yerr=d["val_acc_std"],
            color=ACTIVATION_COLORS[act],
            marker="o",
            capsize=3,
            label=ACTIVATION_LABELS[act],
        )
    ax.axhline(98.0, color=_MUTED, lw=1, ls="--")
    ax.set_xticks(x, [f"{a}\n{params[a] / 1000:.0f} mil parám." for a in archs])
    ax.set_xlabel("Capas ocultas (ordenadas por cantidad de parámetros)")
    ax.set_ylabel("Accuracy de validación (%)")
    ax.set_title(f"Arquitectura × activación · Adam η = {lr:g} · {AUG_LABELS[AUG]} · 3 semillas")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save_figure(fig, out_dir / "E3-G4_arch_activation")


def plot_augmentation(
    grid: pd.DataFrame, activation: str, lr: float, arch: str, out_dir: Path
) -> list[Path]:
    mask = (grid["activation"] == activation) & np.isclose(grid["lr"], lr) & (grid["arch"] == arch)
    sub = grid[mask].set_index("aug").sort_index()
    fig, (ax_gap, ax_rec) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(sub))
    labels = [AUG_LABELS[a] for a in sub.index]
    for col, name, color, dx in (
        ("train_acc", "train", _MUTED, -0.18),
        ("val_acc", "validación", PALETTE[0], 0.18),
    ):
        ax_gap.bar(
            x + dx, sub[f"{col}_mean"], 0.34, yerr=sub[f"{col}_std"], color=color, capsize=3,
            label=name,
        )  # fmt: skip
    for xi, (_, r) in zip(x, sub.iterrows(), strict=True):
        gap = r["train_acc_mean"] - r["val_acc_mean"]
        ax_gap.annotate(f"brecha {gap:.2f}", (xi, 100.2), ha="center", fontsize=11)
    ax_gap.set_ylim(95, 101)
    ax_gap.set_xticks(x, labels)
    ax_gap.set_ylabel("Accuracy (%)")
    ax_gap.set_title("Train vs validación")
    ax_gap.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    for i, digit in enumerate(RARE_DIGITS):
        dx = (i - 0.5) * 0.36
        col = f"recall_{digit}"
        ax_rec.bar(
            x + dx, sub[f"{col}_mean"], 0.34, yerr=sub[f"{col}_std"], color=PALETTE[1 + 2 * i],
            capsize=3, label=f"dígito {digit}",
        )  # fmt: skip
        for xi, v in zip(x, sub[f"{col}_mean"], strict=True):
            ax_rec.annotate(
                f"{v:.1f}", (xi + dx, 81), ha="center", fontsize=11, color="white",
                fontweight="bold",
            )  # fmt: skip
    ax_rec.set_ylim(80, 100)
    ax_rec.set_xticks(x, labels)
    ax_rec.set_ylabel("Recall en validación (%)")
    ax_rec.set_title("Clases escasas: 5 (785 muestras) y 8 (585)")
    ax_rec.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    for ax in (ax_gap, ax_rec):
        ax.grid(axis="x", visible=False)
    act = ACTIVATION_LABELS[activation]
    title = f"Augmentation · {act} · {arch} · Adam η = {lr:g} · búsqueda anterior · 3 semillas"
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    return save_figure(fig, out_dir / "E3-G5_augmentation")


def per_class_test(results_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Recall medio por dígito en test (en %) y cantidad de muestras por dígito."""
    models = _read_json(results_dir / name / "final_eval.json")["models"]
    per_class = [m["test"]["per_class"] for m in models]
    recall = 100.0 * np.mean([p["recall"] for p in per_class], axis=0)
    return recall, np.array(per_class[0]["support"])


def decompose_test_gain(results_dir: Path, before: str, after: str) -> pd.DataFrame:
    """Aporte exacto de cada dígito a la mejora de accuracy en test (Δacc = Σ n_c/N · Δrecall_c)."""
    rec_before, n = per_class_test(results_dir, before)
    rec_after, n_after = per_class_test(results_dir, after)
    if not np.array_equal(n, n_after):
        raise ValueError(f"{before} y {after} no evaluaron el mismo test")
    return pd.DataFrame(
        {
            "digit": np.arange(len(n)),
            "n": n,
            "recall_before": rec_before,
            "recall_after": rec_after,
            "contribution": n / n.sum() * (rec_after - rec_before),
        }
    )


def plot_test_confusion(results_dir: Path, out_dir: Path) -> list[Path]:
    report = _read_json(results_dir / FINAL / "final_eval.json")
    cm = np.mean([m["test"]["confusion_matrix"] for m in report["models"]], axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Muestras (media)")
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xlabel("Predicción del modelo")
    ax.set_ylabel("Etiqueta verdadera")
    mean, std = 100 * report["mean"]["accuracy"], 100 * report["std"]["accuracy"]
    ax.set_title(f"Test · media de {len(report['models'])} modelos: {mean:.2f} ± {std:.2f} %")
    for i in range(10):
        for j in range(10):
            v = cm[i, j]
            txt = f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}"
            color = "white" if v > cm.max() / 2 else "#0b0b0b"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8)
    ax.grid(False)
    fig.tight_layout()
    return save_figure(fig, out_dir / "E3-G6_test_confusion")


def plot_per_class_test(gain: pd.DataFrame, out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(11, 5))
    x = gain["digit"].to_numpy()
    w = 0.38
    ax.bar(x - w / 2, gain["recall_before"], w, color=PALETTE[1], label="Ej2 (digits.csv)")
    ax.bar(x + w / 2, gain["recall_after"], w, color=PALETTE[0], label="Ej3 (unión)")
    for d in RARE_DIGITS:
        r = gain.loc[d]
        for dx, v in ((-w / 2, r["recall_before"]), (w / 2, r["recall_after"])):
            ax.annotate(f"{v:.1f}", (d + dx, v + 1.5), ha="center", fontsize=10)
    ax.set_xticks(x, [str(d) for d in x])
    ax.set_xlabel("Dígito")
    ax.set_ylabel("Recall en test (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Recall por dígito en digits_test.csv · modelo final de cada ejercicio")
    ax.legend(loc="lower left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save_figure(fig, out_dir / "E3-G7_per_class_test")


def plot_hard_examples(results_dir: Path, test_path: Path, out_dir: Path) -> list[Path]:
    report = _read_json(results_dir / FINAL / "final_eval.json")
    best = max(report["models"], key=lambda m: m["test"]["accuracy"])
    seed = best["label"]
    with np.load(results_dir / FINAL / f"final_{seed}" / "test_predictions.npz") as preds:
        y_true, y_pred = preds["labels_true"], preds["labels_pred"]
    X = load_digits_csv(test_path).X
    wrong = np.flatnonzero(y_true != y_pred)[:10]
    fig, axes = plt.subplots(2, 5, figsize=(9, 4.4))
    for ax in axes.ravel():
        ax.axis("off")
    for idx, ax in zip(wrong, axes.ravel(), strict=False):
        ax.imshow(X[idx].reshape(28, 28), cmap="gray_r", interpolation="nearest")
        ax.set_title(f"Real {y_true[idx]} → pred. {y_pred[idx]}", fontsize=10)
    acc = 100 * best["test"]["accuracy"]
    fig.suptitle(f"Errores en test del mejor modelo (semilla {seed.lstrip('s')}, {acc:.2f} %)")
    fig.tight_layout()
    return save_figure(fig, out_dir / "E3-G8_hard_examples")


def main() -> int:
    parser = argparse.ArgumentParser(description="Figuras de la grilla del Ej3")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures/ej3/grid"))
    parser.add_argument("--test-path", type=Path, default=Path("datasets/digits_test.csv"))
    args = parser.parse_args()
    apply_style()

    runs = load_grid_runs(args.results_dir, GRID)
    in_grid = np.isclose(runs["lr"].to_numpy()[:, None], LRS).any(axis=1) & (runs["aug"] == AUG)
    grid = summarize(runs[in_grid])
    winner = grid.iloc[0]
    effects = factor_effects(grid)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.out_dir / "ranking.csv", index=False)
    effects.to_csv(args.out_dir / "efectos.csv", index=False)

    lr, arch = float(winner["lr"]), winner["arch"]
    plot_ranking(grid, args.out_dir)
    plot_factor_effect(effects, args.out_dir)
    plot_activation_lr(grid, arch, AUG, args.out_dir)
    plot_arch_activation(grid, lr, args.out_dir)
    search = summarize(load_grid_runs(args.results_dir, SEARCH))
    cfg = SEARCH_CONFIG
    plot_augmentation(search, cfg["activation"], cfg["lr"], cfg["arch"], args.out_dir)
    acc = f"{winner['val_acc_mean']:.2f} ± {winner['val_acc_std']:.2f}"
    print(f"Mejor: {config_label(winner)} → {acc} ({int(winner['n_seeds'])} semillas)")
    print(effects.round(2).to_string(index=False))

    if (args.results_dir / FINAL / "final_eval.json").exists():
        plot_test_confusion(args.results_dir, args.out_dir)
        plot_hard_examples(args.results_dir, args.test_path, args.out_dir)
        if (args.results_dir / EJ2_FINAL / "final_eval.json").exists():
            gain = decompose_test_gain(args.results_dir, EJ2_FINAL, FINAL)
            gain.to_csv(args.out_dir / "descomposicion.csv", index=False)
            plot_per_class_test(gain, args.out_dir)
            print(gain.round(2).to_string(index=False))
    print(f"Figuras en {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
