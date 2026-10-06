"""Ej2 · Grilla completa optimizador × η × arquitectura × augmentation, y activaciones.

Lee results/ej2_combo (12 optimizador-η × 5 redes × 3 augmentation, tanh) y
results/ej2_activation (3 activaciones × 6 redes × 3 η de Adam, shift ±2). Solo lee
results/ (CLAUDE.md §2.6). Figuras en figures/ej2/grid/:

- E2-G1_ranking: las 10 mejores configuraciones de la grilla (accuracy de validación).
- E2-G2_factor_effect: cuánto mueve la accuracy cada factor (mejor config por nivel).
- E2-G3_optimizer_lr: accuracy y épocas vs η por optimizador, con la red y el shift de la mejor.
- E2-G4_arch_aug: accuracy por red y augmentation, con el optimizador de la mejor.
- E2-G5_activation: mejor η de cada activación por red (Adam, shift ±2).
- E2-G6_augmentation: train vs validación y recall del 5 por augmentation, en la mejor config.
- E2-05_test_confusion_matrix, E2-06_misclassified_test: test de la ganadora
  (results/ej2_final_grid).

Uso:
    python -m analysis.ej2_grid [--results-dir results] [--out-dir figures/ej2/grid]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.common import PALETTE, apply_style, save_figure
from analysis.ej2 import plot_misclassified_test, plot_test_confusion_matrix

COMBO = "ej2_combo"
FINAL = "ej2_final_grid"
ACTIVATION = "ej2_activation"
OPTIMIZERS = ("adam", "rmsprop", "momentum", "gd")
OPTIMIZER_LABELS = {"adam": "Adam", "rmsprop": "RMSProp", "momentum": "Momentum", "gd": "SGD"}
# Mismos colores que analysis.ej2 (E2-02b): el color sigue al optimizador.
OPTIMIZER_COLORS = {
    "adam": PALETTE[0],
    "gd": PALETTE[1],
    "momentum": PALETTE[2],
    "rmsprop": PALETTE[3],
}
AUG_LABELS = {0: "sin shift", 1: "shift ±1 px", 2: "shift ±2 px"}
AUG_COLORS = {0: "#898781", 1: PALETTE[1], 2: PALETTE[0]}
ACTIVATIONS = ("tanh", "relu", "sigmoid")
ACTIVATION_LABELS = {"tanh": "tanh (Xavier)", "relu": "ReLU (He)", "sigmoid": "sigmoide (Xavier)"}
ACTIVATION_COLORS = {"tanh": PALETTE[0], "relu": PALETTE[1], "sigmoid": PALETTE[2]}
# Redes ordenadas por cantidad de parámetros.
ARCH_ORDER = ("[64]", "[128]", "[128, 64]", "[128, 64, 32]", "[256]", "[256, 128]")
DIGIT = 5  # la clase escasa de digits.csv
_MUTED = "#898781"
_LEVEL_LABELS = {
    "aug": lambda v: AUG_LABELS[v],
    "arch": str,
    "optimizer": lambda v: OPTIMIZER_LABELS[v],
    "activation": lambda v: ACTIVATION_LABELS[v].split(" (")[0],
}


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_grid_runs(results_dir: Path, name: str) -> pd.DataFrame:
    """Una fila por corrida de results/<name>: factores de la config y métricas finales."""
    rows = []
    for metrics_path in sorted((results_dir / name).glob("*/metrics.json")):
        m = _read_json(metrics_path)
        c = _read_json(metrics_path.with_name("config.json"))
        training, model = c["training"], c["model"]
        aug = training.get("augmentation") or {}
        rows.append(
            {
                "hash": m["hash"],
                "seed": m["seed"],
                "status": m["status"],
                "optimizer": training["optimizer"]["kind"],
                "lr": training["optimizer"]["lr"],
                "arch": "[" + ", ".join(str(n) for n in model["layers"][1:-1]) + "]",
                "aug": int(aug.get("max_px", 0)) if aug.get("kind") == "random_shift" else 0,
                "activation": model["hidden_activation"],
                "n_params": m["n_params"],
                "epochs": m["epochs_trained"],
                "val_acc": 100.0 * m["val"]["accuracy"],
                "train_acc": 100.0 * m["train"]["accuracy"],
                "recall_digit": 100.0 * m["val"]["per_class"]["recall"][DIGIT],
                **{f"recall_{c}": 100.0 * r for c, r in enumerate(m["val"]["per_class"]["recall"])},
            }
        )
    if not rows:
        raise FileNotFoundError(f"No hay corridas en {results_dir / name}")
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Media y desvío entre semillas por configuración (columnas <métrica>_mean/_std)."""
    keys = ["hash", "optimizer", "lr", "arch", "aug", "activation", "n_params"]
    metrics = ["val_acc", "train_acc", "epochs", *[c for c in runs if c.startswith("recall_")]]
    g = runs[runs["status"] == "ok"].groupby(keys)[metrics]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std"))
    out["n_seeds"] = g.size()
    return out.reset_index().sort_values("val_acc_mean", ascending=False, ignore_index=True)


def config_label(row: pd.Series) -> str:
    opt = OPTIMIZER_LABELS[row["optimizer"]]
    return f"{opt} η = {row['lr']:g} · {row['arch']} · {AUG_LABELS[row['aug']]}"


def factor_effects(combo: pd.DataFrame, activation: pd.DataFrame) -> pd.DataFrame:
    """Para cada factor, la mejor accuracy de cada nivel (optimizando el resto) y su rango."""
    rows = []
    for factor, label, df in (
        ("aug", "Augmentation", combo),
        ("arch", "Arquitectura", combo),
        ("optimizer", "Optimizador (con su mejor η)", combo),
        ("activation", "Activación (con su mejor η)", activation),
    ):
        best = df.groupby(factor)["val_acc_mean"].max()
        best.index = [_LEVEL_LABELS[factor](v) for v in best.index]
        rows.append(
            {
                "factor": label,
                "best_level": best.idxmax(),
                "worst_level": best.idxmin(),
                "best": best.max(),
                "worst": best.min(),
                "range": best.max() - best.min(),
            }
        )
    return pd.DataFrame(rows).sort_values("range", ascending=False, ignore_index=True)


def plot_ranking(combo: pd.DataFrame, out_dir: Path, top: int = 10) -> list[Path]:
    best = combo.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(best))
    colors = [OPTIMIZER_COLORS[o] for o in best["optimizer"]]
    ax.errorbar(
        best["val_acc_mean"],
        y,
        xerr=best["val_acc_std"],
        fmt="none",
        ecolor=_MUTED,
        capsize=4,
        lw=1.5,
    )
    ax.scatter(best["val_acc_mean"], y, c=colors, s=80, zorder=3, edgecolors="white", linewidths=2)
    for yi, (_, r) in zip(y, best.iterrows(), strict=True):
        ax.annotate(
            f"{r['val_acc_mean']:.2f} ± {r['val_acc_std']:.2f}",
            (r["val_acc_mean"] + r["val_acc_std"], yi),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=11,
        )
    ax.set_yticks(y, [config_label(r) for _, r in best.iterrows()], fontsize=12)
    ax.set_xlabel("Accuracy de validación (%)")
    ax.set_title(f"Las {top} mejores de {len(combo)} configuraciones (tanh, 3 semillas)")
    ax.grid(axis="y", visible=False)
    lo = (best["val_acc_mean"] - best["val_acc_std"]).min()
    hi = (best["val_acc_mean"] + best["val_acc_std"]).max()
    ax.set_xlim(lo - 0.1, hi + 0.35)
    fig.tight_layout()
    return save_figure(fig, out_dir / "E2-G1_ranking")


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
    return save_figure(fig, out_dir / "E2-G2_factor_effect")


def plot_optimizer_lr(combo: pd.DataFrame, arch: str, aug: int, out_dir: Path) -> list[Path]:
    sub = combo[(combo["arch"] == arch) & (combo["aug"] == aug)]
    fig, (ax_acc, ax_ep) = plt.subplots(1, 2, figsize=(13, 5))
    for opt in OPTIMIZERS:
        d = sub[sub["optimizer"] == opt].sort_values("lr")
        kw = {
            "color": OPTIMIZER_COLORS[opt],
            "marker": "o",
            "capsize": 3,
            "label": OPTIMIZER_LABELS[opt],
        }
        ax_acc.errorbar(d["lr"], d["val_acc_mean"], yerr=d["val_acc_std"], **kw)
        ax_ep.errorbar(d["lr"], d["epochs_mean"], yerr=d["epochs_std"], **kw)
    for ax in (ax_acc, ax_ep):
        ax.set_xscale("log")
        ax.set_xlabel("Tasa de aprendizaje η (escala log)")
    ax_acc.set_ylabel("Accuracy de validación (%)")
    ax_acc.set_title("Accuracy final")
    ax_ep.set_ylabel("Épocas hasta early stopping")
    ax_ep.set_title("Velocidad")
    handles, labels = ax_acc.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(OPTIMIZERS), bbox_to_anchor=(0.5, -0.06)
    )
    title = f"Optimizador y η · red {arch} · {AUG_LABELS[aug]} · tanh · 3 semillas"
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    return save_figure(fig, out_dir / "E2-G3_optimizer_lr")


def plot_arch_aug(combo: pd.DataFrame, optimizer: str, lr: float, out_dir: Path) -> list[Path]:
    sub = combo[(combo["optimizer"] == optimizer) & np.isclose(combo["lr"], lr)]
    archs = [a for a in ARCH_ORDER if a in set(sub["arch"])]
    params = sub.groupby("arch")["n_params"].first()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(archs))
    for aug in sorted(AUG_LABELS):
        d = sub[sub["aug"] == aug].set_index("arch").loc[archs]
        ax.errorbar(
            x,
            d["val_acc_mean"],
            yerr=d["val_acc_std"],
            color=AUG_COLORS[aug],
            marker="o",
            capsize=3,
            label=AUG_LABELS[aug],
        )
    ax.set_xticks(x, [f"{a}\n{params[a] / 1000:.0f} mil parám." for a in archs])
    ax.set_xlabel("Capas ocultas (ordenadas por cantidad de parámetros)")
    ax.set_ylabel("Accuracy de validación (%)")
    opt = OPTIMIZER_LABELS[optimizer]
    ax.set_title(f"Arquitectura × augmentation · {opt} η = {lr:g} · tanh · 3 semillas")
    ax.legend(loc="center left")
    fig.tight_layout()
    return save_figure(fig, out_dir / "E2-G4_arch_aug")


def plot_activation(activation: pd.DataFrame, out_dir: Path) -> list[Path]:
    best = activation.loc[activation.groupby(["arch", "activation"])["val_acc_mean"].idxmax()]
    archs = [a for a in ARCH_ORDER if a in set(best["arch"])]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(archs))
    width = 0.22
    for i, act in enumerate(ACTIVATIONS):
        d = best[best["activation"] == act].set_index("arch").loc[archs]
        ax.errorbar(
            x + (i - 1) * width,
            d["val_acc_mean"],
            yerr=d["val_acc_std"],
            fmt="o",
            color=ACTIVATION_COLORS[act],
            capsize=3,
            label=ACTIVATION_LABELS[act],
        )
    ax.set_xticks(x, archs)
    ax.set_xlabel("Capas ocultas")
    ax.set_ylabel("Accuracy de validación (%)")
    ax.set_title("Activación oculta con su mejor η · Adam · shift ±2 px · 3 semillas")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save_figure(fig, out_dir / "E2-G5_activation")


def plot_augmentation(
    combo: pd.DataFrame, optimizer: str, lr: float, arch: str, out_dir: Path
) -> list[Path]:
    sub = combo[
        (combo["optimizer"] == optimizer) & np.isclose(combo["lr"], lr) & (combo["arch"] == arch)
    ]
    sub = sub.set_index("aug").sort_index()
    fig, (ax_gap, ax_rec) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(sub))
    labels = [AUG_LABELS[a] for a in sub.index]
    for col, name, color, dx in (
        ("train_acc", "train", _MUTED, -0.18),
        ("val_acc", "validación", PALETTE[0], 0.18),
    ):
        ax_gap.bar(
            x + dx,
            sub[f"{col}_mean"],
            0.34,
            yerr=sub[f"{col}_std"],
            color=color,
            capsize=3,
            label=name,
        )
    for xi, (_, r) in zip(x, sub.iterrows(), strict=True):
        ax_gap.annotate(
            f"brecha {r['train_acc_mean'] - r['val_acc_mean']:.2f}",
            (xi, 100.2),
            ha="center",
            fontsize=11,
        )
    ax_gap.set_ylim(95, 101)
    ax_gap.set_xticks(x, labels)
    ax_gap.set_ylabel("Accuracy (%)")
    ax_gap.set_title("Train vs validación")
    ax_gap.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax_rec.bar(
        x, sub["recall_digit_mean"], 0.5, yerr=sub["recall_digit_std"], color=PALETTE[1], capsize=3
    )
    for xi, (_, r) in zip(x, sub.iterrows(), strict=True):
        ax_rec.annotate(
            f"{r['recall_digit_mean']:.1f} %",
            (xi, 72),
            ha="center",
            fontsize=12,
            color="white",
            fontweight="bold",
        )
    ax_rec.set_ylim(70, 100)
    ax_rec.set_xticks(x, labels)
    ax_rec.set_ylabel(f"Recall del dígito {DIGIT} en validación (%)")
    ax_rec.set_title(f"Dígito {DIGIT} (271 muestras en digits.csv)")
    for ax in (ax_gap, ax_rec):
        ax.grid(axis="x", visible=False)
    opt = OPTIMIZER_LABELS[optimizer]
    fig.suptitle(f"Augmentation · {opt} η = {lr:g} · {arch} · tanh · 3 semillas", fontsize=15)
    fig.tight_layout()
    return save_figure(fig, out_dir / "E2-G6_augmentation")


def main() -> int:
    parser = argparse.ArgumentParser(description="Figuras de la grilla completa del Ej2")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures/ej2/grid"))
    parser.add_argument("--test-path", type=Path, default=Path("datasets/digits_test.csv"))
    args = parser.parse_args()
    apply_style()

    combo = summarize(load_grid_runs(args.results_dir, COMBO))
    activation = summarize(load_grid_runs(args.results_dir, ACTIVATION))
    winner = combo.iloc[0]
    effects = factor_effects(combo, activation)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    combo.to_csv(args.out_dir / "ranking.csv", index=False)
    effects.to_csv(args.out_dir / "efectos.csv", index=False)
    plot_ranking(combo, args.out_dir)
    plot_factor_effect(effects, args.out_dir)
    plot_optimizer_lr(combo, winner["arch"], int(winner["aug"]), args.out_dir)
    plot_arch_aug(combo, winner["optimizer"], float(winner["lr"]), args.out_dir)
    plot_activation(activation, args.out_dir)
    plot_augmentation(combo, winner["optimizer"], float(winner["lr"]), winner["arch"], args.out_dir)
    if (args.results_dir / FINAL / "final_eval.json").exists():
        plot_test_confusion_matrix(args.results_dir, args.out_dir, final_run=FINAL)
        plot_misclassified_test(args.results_dir, args.test_path, args.out_dir, final_run=FINAL)

    acc = f"{winner['val_acc_mean']:.2f} ± {winner['val_acc_std']:.2f}"
    print(f"Mejor: {config_label(winner)} → {acc}")
    print(effects.round(2).to_string(index=False))
    print(f"Figuras en {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
