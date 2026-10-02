"""Ej3 · More Digits: análisis cuantitativo, ablación y figuras de producción.

Genera las figuras y tablas para docs/resultados/ej3.md:
- E3-01_accuracy_progression: Progresión de exactitud con cota 98%.
- E3-02_ablation_ranking: Tabla/gráfico de ablación de técnicas.
- E3-03_per_class_recall_comparison: Recall por dígito (impacto en 8 y 5).
- E3-04_test_confusion_matrix: Matriz de confusión oficial en test.
- E3-05_hard_examples: Muestras difíciles del test set mal clasificadas.

Uso:
    python -m analysis.ej3 [--results-dir results] [--out-dir figures/ej3]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.common import PALETTE, apply_style, save_figure
from data.loaders import load_digits_csv


def plot_accuracy_progression(results_dir: Path, out_dir: Path) -> Path:
    """Progresión de exactitud desde Ej2 hasta el récord de Ej3 en test."""
    apply_style()
    with open(results_dir / "ej2_final" / "final_eval.json") as f:
        ej2_final = json.load(f)

    df_base = pd.read_csv(results_dir / "ej3_baseline" / "summary.csv")
    df_search = pd.read_csv(results_dir / "ej3_search" / "summary.csv")
    df_best = pd.read_csv(results_dir / "ej3_best" / "summary.csv")

    with open(results_dir / "ej3_final" / "final_eval.json") as f:
        ej3_final = json.load(f)

    # Mejor config en búsqueda
    best_search_acc = df_search["val.accuracy_mean"].max() * 100.0
    best_search_std = (
        df_search.loc[df_search["val.accuracy_mean"].idxmax(), "val.accuracy_std"] * 100.0
    )

    milestones = [
        (
            "Ej2 Final\n(Test sin 8s)",
            ej2_final["mean"]["accuracy"] * 100.0,
            ej2_final["std"]["accuracy"] * 100.0,
            PALETTE[7],
        ),
        (
            "Ej3 Base\n(Dato unión)",
            df_base["val.accuracy_mean"].values[0] * 100.0,
            df_base["val.accuracy_std"].values[0] * 100.0,
            PALETTE[3],
        ),
        ("Ej3 Búsqueda\n(ReLU+Shift)", best_search_acc, best_search_std, PALETTE[0]),
        (
            "Ej3 Best\n(Val 5 seeds)",
            df_best["val.accuracy_mean"].values[0] * 100.0,
            df_best["val.accuracy_std"].values[0] * 100.0,
            PALETTE[2],
        ),
        (
            "Ej3 Final\n(Test Sagrado)",
            ej3_final["mean"]["accuracy"] * 100.0,
            ej3_final["std"]["accuracy"] * 100.0,
            PALETTE[5],
        ),
    ]

    labels, accs, errs, colors = zip(*milestones, strict=True)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.bar(
        x, accs, yerr=errs, capsize=4, color=colors, width=0.55, edgecolor="#0b0b0b", lw=0.8
    )

    ax.axhline(98.0, color="#e34948", linestyle="--", lw=1.8, label="Meta Requerida (≥ 98 %)")
    ax.axhline(
        90.27, color="#898781", linestyle=":", lw=1.2, label="Cota Teórica Ej2 (sin 8s: 90.27%)"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(80, 101)
    ax.set_ylabel("Exactitud (%)")
    ax.set_title("Evolución del Desempeño: de Ej2 a la Meta de Ej3 (≥ 98%)")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(frameon=True, loc="lower right")

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1.0,
            f"{height:.2f}%\n±{errs[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            weight="bold",
        )

    fig.tight_layout()
    return save_figure(fig, out_dir / "E3-01_accuracy_progression")


def plot_ablation_ranking(results_dir: Path, out_dir: Path) -> Path:
    """Ablación de factores en Ej3 (ordenados por aporte sobre la línea base)."""
    apply_style()
    base_val = 96.92  # ej3_baseline

    # Factores aislados
    items = [
        ("Base (Tanh 128, sin aug)", base_val, 0.44),
        ("+ ReLU (en lugar de Tanh)", 97.19, 0.57),
        ("+ Capacidad (ReLU [256, 128])", 97.65, 0.52),
        ("+ RandomShift ±1 px (Tanh 128)", 98.11, 0.43),
        ("+ ReLU 128 + RandomShift ±1 px", 98.30, 0.36),
        ("+ Combinación Óptima (ReLU [256, 128] + Shift ±1)", 98.48, 0.24),
    ]

    labels = [it[0] for it in items]
    vals = [it[1] for it in items]
    errs = [it[2] for it in items]
    diffs = [v - base_val for v in vals]

    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(
        y, diffs, xerr=errs, capsize=3, color=PALETTE[0], height=0.55, edgecolor="#0b0b0b", lw=0.8
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Ganancia de Exactitud respecto a la Base de Ej3 (+ puntos %)")
    ax.set_title("Estudio de Ablación: Contribución de Cada Técnica")
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")

    for i, bar in enumerate(bars):
        w = bar.get_width()
        text = f"+{w:.2f}% ({vals[i]:.2f}%)" if w > 0 else f"{vals[i]:.2f}%"
        ax.text(
            w + 0.08,
            bar.get_y() + bar.get_height() / 2.0,
            text,
            va="center",
            fontsize=8,
            weight="bold",
        )

    ax.set_xlim(-0.2, 2.2)
    fig.tight_layout()
    return save_figure(fig, out_dir / "E3-02_ablation_ranking")


def plot_per_class_recall_comparison(results_dir: Path, out_dir: Path) -> Path:
    """Comparativa de recall por dígito entre Ej2 Final y Ej3 Final en test."""
    apply_style()
    with open(results_dir / "ej2_final" / "final_eval.json") as f:
        ej2 = json.load(f)
    with open(results_dir / "ej3_final" / "final_eval.json") as f:
        ej3 = json.load(f)

    rec_ej2 = (
        np.mean(
            [[m["test"]["per_class"]["recall"][c] for m in ej2["models"]] for c in range(10)],
            axis=1,
        )
        * 100.0
    )
    rec_ej3 = (
        np.mean(
            [[m["test"]["per_class"]["recall"][c] for m in ej3["models"]] for c in range(10)],
            axis=1,
        )
        * 100.0
    )

    digits = np.arange(10)
    w = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(
        digits - w / 2,
        rec_ej2,
        w,
        label="Ej2 Final (Falta de 8s en train)",
        color=PALETTE[1],
        edgecolor="#0b0b0b",
        lw=0.8,
    )
    ax.bar(
        digits + w / 2,
        rec_ej3,
        w,
        label="Ej3 Final (Unión deduplicada + ReLU + Shift)",
        color=PALETTE[0],
        edgecolor="#0b0b0b",
        lw=0.8,
    )

    ax.set_xticks(digits)
    ax.set_xticklabels([f"Dígito {d}" for d in digits], fontsize=8.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Sensibilidad / Recall en Test (%)")
    ax.set_title("Impacto del Dataset y Técnicas por Dígito: Ej2 vs Ej3 en Producción")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(frameon=True, loc="lower left")

    # Resaltar 8 y 5
    for d in [5, 8]:
        ax.text(
            d - w / 2,
            rec_ej2[d] + 2.0,
            f"{rec_ej2[d]:.1f}%",
            ha="center",
            fontsize=7.5,
            weight="bold",
            color="#eb6834",
        )
        ax.text(
            d + w / 2,
            rec_ej3[d] + 2.0,
            f"{rec_ej3[d]:.1f}%",
            ha="center",
            fontsize=7.5,
            weight="bold",
            color="#2a78d6",
        )

    fig.tight_layout()
    return save_figure(fig, out_dir / "E3-03_per_class_recall_comparison")


def plot_test_confusion_matrix(results_dir: Path, out_dir: Path) -> Path:
    """Matriz de confusión oficial de Ej3 Final en digits_test.csv."""
    apply_style()
    with open(results_dir / "ej3_final" / "final_eval.json") as f:
        report = json.load(f)

    # Promediar matrices de confusión entre semillas
    cms = [np.array(m["test"]["confusion_matrix"]) for m in report["models"]]
    cm_mean = np.mean(cms, axis=0)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_mean, cmap="Blues", interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Muestras promedio")

    n_classes = 10
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xlabel("Predicción del Modelo")
    ax.set_ylabel("Etiqueta Verdadera")
    acc_str = f"{report['mean']['accuracy'] * 100:.2f}%"
    ax.set_title(f"Matriz de Confusión Oficial en Test (Ej3 Final, Acc = {acc_str})")

    thresh = cm_mean.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm_mean[i, j]
            val_str = f"{val:.0f}" if val.is_integer() else f"{val:.1f}"
            color = "white" if val > thresh else "#0b0b0b"
            ax.text(j, i, val_str, ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    return save_figure(fig, out_dir / "E3-04_test_confusion_matrix")


def plot_hard_examples(results_dir: Path, test_path: Path, out_dir: Path) -> Path:
    """Grilla de las pocas muestras de test que el modelo final falló."""
    apply_style()
    with np.load(results_dir / "ej3_final" / "final_s1" / "test_predictions.npz") as preds:
        y_true = preds["labels_true"]
        y_pred = preds["labels_pred"]

    dataset = load_digits_csv(test_path)
    X_test = dataset.X

    wrong_idx = np.flatnonzero(y_true != y_pred)
    selected = wrong_idx[:10]  # Primeros 10 errores de la mejor semilla

    fig, axes = plt.subplots(2, 5, figsize=(9, 4.2))
    for idx, ax in zip(selected, axes.ravel(), strict=True):
        img = X_test[idx].reshape(28, 28)
        ax.imshow(img, cmap="gray_r", interpolation="nearest")
        true_l = y_true[idx]
        pred_l = y_pred[idx]
        ax.set_title(f"Real: {true_l} → Pred: {pred_l}", fontsize=8, color="#e34948", weight="bold")
        ax.axis("off")

    fig.suptitle(
        "Ejemplos Ambiguos de digits_test.csv Mal Clasificados por Ej3 (Semilla 1)",
        fontsize=10,
        weight="bold",
    )
    fig.tight_layout()
    return save_figure(fig, out_dir / "E3-05_hard_examples")


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera figuras cuantitativas para Ej3")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--test-path", type=Path, default=Path("datasets/digits_test.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures/ej3"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generando figuras para Ej3 en {args.out_dir}...")

    plot_accuracy_progression(args.results_dir, args.out_dir)
    print("✓ E3-01_accuracy_progression")

    plot_ablation_ranking(args.results_dir, args.out_dir)
    print("✓ E3-02_ablation_ranking")

    plot_per_class_recall_comparison(args.results_dir, args.out_dir)
    print("✓ E3-03_per_class_recall_comparison")

    plot_test_confusion_matrix(args.results_dir, args.out_dir)
    print("✓ E3-04_test_confusion_matrix")

    plot_hard_examples(args.results_dir, args.test_path, args.out_dir)
    print("✓ E3-05_hard_examples")

    print("Todas las figuras de Ej3 generadas exitosamente.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
