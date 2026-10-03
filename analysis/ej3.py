"""Ej3 · More Digits: análisis cuantitativo, ablación y figuras de producción.

Genera las figuras y tablas para docs/resultados/ej3.md:
- E3-01_accuracy_progression: Progresión de exactitud con cota 98%.
- E3-02_ablation_ranking: Grilla de ablación en validación (red × RandomShift).
- E3-03_per_class_recall_comparison: Recall por dígito (impacto en 8 y 5).
- E3-04_test_confusion_matrix: Matriz de confusión oficial en test.
- E3-05_hard_examples: Muestras difíciles del test set mal clasificadas.

Además imprime la descomposición por dígito de la mejora en test de Ej2 a Ej3
(decompose_test_gain).

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

# Grilla de ablación de E3-02 sobre ej3_search, en el orden fijo de la figura:
# filas (etiqueta, activación oculta, capas) y columnas (etiqueta, max_px del RandomShift).
ABLATION_NETWORKS = (
    ("tanh [128]", "tanh", [784, 128, 10]),
    ("ReLU [128]", "relu", [784, 128, 10]),
    ("ReLU [256, 128]", "relu", [784, 256, 128, 10]),
)
ABLATION_SHIFTS = (("Sin shift", None), ("Shift ±1 px", 1.0), ("Shift ±2 px", 2.0))


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

    # Cada etiqueta dice sobre qué conjunto se midió la barra: validación y test no se comparan.
    milestones = [
        (
            "Ej2 Final\nTEST\n(entrenado sin 8s)",
            ej2_final["mean"]["accuracy"] * 100.0,
            ej2_final["std"]["accuracy"] * 100.0,
            PALETTE[7],
        ),
        (
            "Ej3 Base\nVALIDACIÓN\n(dato unión)",
            df_base["val.accuracy_mean"].values[0] * 100.0,
            df_base["val.accuracy_std"].values[0] * 100.0,
            PALETTE[3],
        ),
        ("Ej3 Búsqueda\nVALIDACIÓN\n(ReLU + shift)", best_search_acc, best_search_std, PALETTE[0]),
        (
            "Ej3 Best\nVALIDACIÓN\n(5 semillas)",
            df_best["val.accuracy_mean"].values[0] * 100.0,
            df_best["val.accuracy_std"].values[0] * 100.0,
            PALETTE[2],
        ),
        (
            "Ej3 Final\nTEST\n(sagrado)",
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


def ablation_grid(results_dir: Path) -> pd.DataFrame:
    """Accuracy de validación de ej3_search para ABLATION_NETWORKS × ABLATION_SHIFTS.

    Una fila por celda, redes por filas y shift por columnas: network, shift,
    hash, n (semillas), accuracy_mean y accuracy_std (ddof = 1), tal como
    están en results/ej3_search/summary.csv. Una celda que no aparezca
    exactamente una vez en el barrido es un ValueError.
    """
    df = pd.read_csv(results_dir / "ej3_search" / "summary.csv")
    layers = df["model.layers"].map(json.loads)
    rows = []
    for network, activation, sizes in ABLATION_NETWORKS:
        is_network = (df["model.hidden_activation"] == activation) & layers.map(
            lambda v, sizes=sizes: v == sizes
        )
        for shift, max_px in ABLATION_SHIFTS:
            if max_px is None:
                is_shift = df["training.augmentation.kind"].isna()
            else:
                is_shift = (df["training.augmentation.kind"] == "random_shift") & (
                    df["training.augmentation.max_px"] == max_px
                )
            cell = df[is_network & is_shift]
            if len(cell) != 1:
                raise ValueError(
                    f"ej3_search: {len(cell)} configuraciones para {network!r} con {shift!r}, "
                    "se esperaba 1"
                )
            row = cell.iloc[0]
            rows.append(
                {
                    "network": network,
                    "shift": shift,
                    "hash": row["hash"],
                    "n": int(row["n"]),
                    "accuracy_mean": row["val.accuracy_mean"],
                    "accuracy_std": row["val.accuracy_std"],
                }
            )
    return pd.DataFrame(rows)


def plot_ablation_ranking(results_dir: Path, out_dir: Path) -> Path:
    """Grilla de ablación de Ej3: accuracy de validación (media ± desvío) por red y RandomShift."""
    apply_style()
    grid = ablation_grid(results_dir)
    networks = [name for name, *_ in ABLATION_NETWORKS]
    shifts = [name for name, _ in ABLATION_SHIFTS]
    shape = (len(networks), len(shifts))
    mean = grid["accuracy_mean"].to_numpy().reshape(shape) * 100.0
    std = grid["accuracy_std"].to_numpy().reshape(shape) * 100.0

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    im = ax.imshow(mean, cmap="Blues", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Exactitud de validación (%)")

    ax.set_xticks(np.arange(len(shifts)))
    ax.set_xticklabels(shifts)
    ax.set_yticks(np.arange(len(networks)))
    ax.set_yticklabels(networks)
    ax.set_xlabel("Data augmentation (RandomShift)")
    ax.set_ylabel("Red (capas ocultas)")
    ax.set_title("Ablación en Validación: Red × RandomShift")
    ax.grid(False)

    thresh = (mean.max() + mean.min()) / 2.0
    best = np.unravel_index(np.argmax(mean), shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            ax.text(
                j,
                i,
                f"{mean[i, j]:.2f}\n± {std[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if mean[i, j] > thresh else "#0b0b0b",
                fontsize=13,
                weight="bold" if (i, j) == best else "normal",
            )

    fig.text(
        0.5,
        0.005,
        f"Unión deduplicada · sigmoide + MSE · Adam η = 0.001 · lote 32 · "
        f"media ± desvío entre {grid['n'].min()} semillas · en negrita, la mejor",
        ha="center",
        fontsize=10,
        style="italic",
        color="#52514e",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save_figure(fig, out_dir / "E3-02_ablation_ranking")


def decompose_test_gain(results_dir: Path) -> pd.DataFrame:
    """Aporte de cada dígito a la mejora de accuracy en test de Ej2 Final a Ej3 Final.

    accuracy = Σ_c (n_c / N) · recall_c, así que
    Δacc = Σ_c (n_c / N) · (recall_ej3_c − recall_ej2_c) es exacta. Una fila
    por dígito: digit, n (muestras de test), recall_ej2 y recall_ej3 (media
    entre los modelos de cada final_eval.json) y contribution (su término de
    la suma, en puntos porcentuales). Solo lee results/ej2_final y
    results/ej3_final; si no evaluaron el mismo test es un ValueError.
    """
    recalls, supports = {}, {}
    for name in ("ej2_final", "ej3_final"):
        with open(results_dir / name / "final_eval.json") as f:
            models = json.load(f)["models"]
        per_class = [m["test"]["per_class"] for m in models]
        if any(p["support"] != per_class[0]["support"] for p in per_class):
            raise ValueError(f"{name}: los modelos no evaluaron el mismo test")
        supports[name] = np.array(per_class[0]["support"])
        recalls[name] = np.mean([p["recall"] for p in per_class], axis=0)
    if not np.array_equal(supports["ej2_final"], supports["ej3_final"]):
        raise ValueError("ej2_final y ej3_final no evaluaron el mismo test")

    n = supports["ej3_final"]
    delta = recalls["ej3_final"] - recalls["ej2_final"]
    return pd.DataFrame(
        {
            "digit": np.arange(len(n)),
            "n": n,
            "recall_ej2": recalls["ej2_final"],
            "recall_ej3": recalls["ej3_final"],
            "contribution": n / n.sum() * delta * 100.0,
        }
    )


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

    gain = decompose_test_gain(args.results_dir)
    eight = gain.loc[gain["digit"] == 8, "contribution"].sum()
    total = gain["contribution"].sum()
    print(
        f"Mejora en test de Ej2 a Ej3: {total:+.2f} puntos = {eight:+.2f} del dígito 8 "
        f"{total - eight:+.2f} del resto"
    )

    print("Todas las figuras de Ej3 generadas exitosamente.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
