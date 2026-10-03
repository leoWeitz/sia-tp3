"""Ej2 · Clasificación de dígitos con MLP: análisis cuantitativo y figuras.

Genera las figuras y tablas para docs/resultados/ej2.md:
- E2-01_evaluation_protocol: Esquema del protocolo de evaluación (F12 Parte 2).
- E2-02_lr_optimizer: Comparativa de lr vs val_loss y val_acc para SGD y Adam.
- E2-02b_optimizers: Mejor η de cada mecanismo de optimización (accuracy de val y épocas).
- E2-03_architecture: Val accuracy vs parámetros (ancho y profundidad).
- E2-04_augmentation_impact: Efecto de data augmentation y recall del dígito 5.
- E2-05_test_confusion_matrix: Matriz de confusión en digits_test.csv.
- E2-06_misclassified_test: Muestras del test erradas por el modelo (falta de 8s).

Uso:
    python -m analysis.ej2 [--results-dir results] [--out-dir figures/ej2]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.common import PALETTE, aggregate, apply_style, load_runs, save_figure
from data.loaders import load_digits_csv

# Mecanismos de optimización de E2-02b, en el orden fijo de la figura y la tabla:
# (etiqueta, experimento, training.optimizer.kind, con η adaptativo).
OPTIMIZER_VARIANTS = (
    ("SGD", "ej2_lr", "gd", False),
    ("Momentum", "ej2_opt", "momentum", False),
    ("RMSProp", "ej2_opt", "rmsprop", False),
    ("Adam", "ej2_lr", "adam", False),
    ("SGD + η adaptativo", "ej2_eta_adapt", "gd", True),
)
# El color sigue al optimizador: Adam y SGD llevan los mismos que en E2-02.
OPTIMIZER_COLORS = {
    "Adam": PALETTE[0],
    "SGD": PALETTE[1],
    "Momentum": PALETTE[2],
    "RMSProp": PALETTE[3],
    "SGD + η adaptativo": PALETTE[4],
}
_OPTIMIZER_METRICS = (
    "val.accuracy",
    "val.loss",
    "val.macro_f1",
    "best_epoch",
    "epochs_trained",
    "lr_final",
)
_ADAPTIVE_ETA_PARAMS = ("a", "b", "k", "k_prime")


def plot_protocol(out_dir: Path) -> Path:
    """Diagrama del protocolo de evaluación estratificado y test sagrado."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis("off")

    # Cajas principales
    boxes = [
        ("digits.csv\n(12 449 muestras)", 0.08, 0.5, 0.22, 0.35, "#e8f0fe", "#2a78d6"),
        (
            "Train Set (80%)\n(9 959 muestras)\nEntrenamiento Backprop",
            0.42,
            0.65,
            0.25,
            0.28,
            "#e6f4ea",
            "#1baf7a",
        ),
        (
            "Val Set (20%)\n(2 490 muestras)\nTuning y Early Stopping",
            0.42,
            0.15,
            0.25,
            0.28,
            "#fef7e0",
            "#eda100",
        ),
        (
            "digits_test.csv\n(2 497 muestras)\nEvaluación Final Una Sola Vez",
            0.76,
            0.4,
            0.22,
            0.4,
            "#fce8e6",
            "#e34948",
        ),
    ]

    from matplotlib.patches import FancyBboxPatch

    for text, x, y, w, h, bg, border in boxes:
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            facecolor=bg,
            edgecolor=border,
            linewidth=1.5,
            boxstyle="round,pad=0.03",
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=9,
            color="#0b0b0b",
            weight="semibold",
            transform=ax.transAxes,
        )

    # Flechas
    arrow_props = dict(arrowstyle="->", lw=1.5, color="#52514e")
    ax.annotate("", xy=(0.42, 0.79), xytext=(0.30, 0.68), arrowprops=arrow_props)
    ax.annotate("", xy=(0.42, 0.29), xytext=(0.30, 0.55), arrowprops=arrow_props)
    ax.annotate(
        "",
        xy=(0.76, 0.60),
        xytext=(0.67, 0.79),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#1baf7a", ls="--"),
    )
    ax.text(0.72, 0.74, "--final-eval", color="#1baf7a", fontsize=8, weight="bold", ha="center")

    ax.text(
        0.5,
        0.02,
        "Protocolo: Stratified Holdout 80/20 con 3 semillas; "
        "Test sagrado aislado hasta --final-eval",
        ha="center",
        fontsize=9,
        style="italic",
        color="#52514e",
        transform=ax.transAxes,
    )

    fig.tight_layout()
    return save_figure(fig, out_dir / "E2-01_evaluation_protocol")


def plot_lr_optimizer(results_dir: Path, out_dir: Path) -> Path:
    """Sensibilidad a la tasa de aprendizaje para SGD y Adam."""
    apply_style()
    df = pd.read_csv(results_dir / "ej2_lr" / "summary.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    for opt_name, color, marker in [("adam", PALETTE[0], "o"), ("gd", PALETTE[1], "s")]:
        sub = df[df["training.optimizer.kind"] == opt_name].sort_values("training.optimizer.lr")
        label = "Adam" if opt_name == "adam" else "SGD Mini-batch"
        lrs = sub["training.optimizer.lr"].values
        acc_m = sub["val.accuracy_mean"].values * 100.0
        acc_s = sub["val.accuracy_std"].values * 100.0
        loss_m = sub["val.loss_mean"].values
        loss_s = sub["val.loss_std"].values

        ax1.errorbar(
            lrs, acc_m, yerr=acc_s, label=label, color=color, marker=marker, capsize=3, lw=1.5
        )
        ax2.errorbar(
            lrs, loss_m, yerr=loss_s, label=label, color=color, marker=marker, capsize=3, lw=1.5
        )

    ax1.set_xscale("log")
    ax1.set_xlabel("Tasa de aprendizaje (η)")
    ax1.set_ylabel("Exactitud de validación (%)")
    ax1.set_title("Exactitud vs Tasa de Aprendizaje")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Tasa de aprendizaje (η)")
    ax2.set_ylabel("Pérdida de validación MSE (log)")
    ax2.set_title("Pérdida MSE vs Tasa de Aprendizaje")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True)

    fig.tight_layout()
    return save_figure(fig, out_dir / "E2-02_lr_optimizer")


def best_per_optimizer(results_dir: Path) -> pd.DataFrame:
    """Mejor configuración de cada mecanismo de optimización (OPTIMIZER_VARIANTS, en ese orden).

    "Mejor" = mayor accuracy media de validación entre las configuraciones del
    mecanismo sin corridas divergidas (si todas divergieron, entre todas).
    Columnas: label, experiment, hash, lr (η inicial), alpha y a/b/k/k_prime
    (NaN donde no aplican), n_configs (configuraciones comparadas), n
    (semillas), n_stopped (corridas que cortó el early stopping) y <m>_mean,
    <m>_std (ddof = 1), <m>_min, <m>_max de val.accuracy, val.loss,
    val.macro_f1, best_epoch, epochs_trained y lr_final.
    """
    rows = []
    for label, experiment, kind, adaptive in OPTIMIZER_VARIANTS:
        runs = load_runs(results_dir / experiment)
        if not runs.empty:
            adaptive_column = "training.adaptive_eta.a"
            has_adaptive = (
                runs[adaptive_column].notna()
                if adaptive_column in runs
                else pd.Series(False, index=runs.index)
            )
            runs = runs[(runs["training.optimizer.kind"] == kind) & (has_adaptive == adaptive)]
        if runs.empty:
            raise FileNotFoundError(
                f"{results_dir / experiment}: no hay corridas terminadas de {label!r}"
            )

        metrics = [m for m in _OPTIMIZER_METRICS if m in runs]
        stats = aggregate(runs, "hash", metrics).set_index("hash")
        groups = runs.groupby("hash")
        stats["n_stopped"] = groups["stopped_epoch"].apply(lambda s: int(s.notna().sum()))
        all_ok = groups["status"].apply(lambda s: bool((s == "ok").all()))
        candidates = stats[all_ok] if all_ok.any() else stats
        best = candidates["val.accuracy_mean"].idxmax()
        config = runs[runs["hash"] == best].iloc[0]
        rows.append(
            {
                "label": label,
                "experiment": experiment,
                "hash": best,
                "lr": config["training.optimizer.lr"],
                "alpha": config.get("training.optimizer.alpha", np.nan),
                **{
                    p: config.get(f"training.adaptive_eta.{p}", np.nan)
                    for p in _ADAPTIVE_ETA_PARAMS
                },
                "n_configs": len(stats),
                **stats.loc[best].to_dict(),
            }
        )
    return pd.DataFrame(rows)


def _optimizer_tick(row: pd.Series) -> str:
    """Rótulo de dos líneas: el mecanismo y los hiperparámetros de su mejor configuración."""
    if not np.isnan(row["a"]):
        detail = (
            f"η₀ = {row['lr']:g}, a = {row['a']:g}, b = {row['b']:g}, "
            f"k = {row['k']:g}, k′ = {row['k_prime']:g}"
        )
    elif not np.isnan(row["alpha"]):
        detail = f"η = {row['lr']:g}, α = {row['alpha']:g}"
    else:
        detail = f"η = {row['lr']:g}"
    return f"{row['label']}\n{detail}"


def plot_optimizers(results_dir: Path, out_dir: Path) -> list[Path]:
    """Accuracy de validación y épocas hasta el corte del mejor η de cada optimizador.

    Dos paneles con los mecanismos en el mismo orden: media ± desvío entre
    semillas de la accuracy de validación y de las épocas entrenadas hasta que
    cortó el early stopping (o hasta agotar las épocas, que se indica).
    """
    apply_style()
    best = best_per_optimizer(results_dir)
    y = np.arange(len(best))[::-1]  # el primer mecanismo, arriba
    colors = [OPTIMIZER_COLORS[label] for label in best["label"]]
    acc_m = best["val.accuracy_mean"].to_numpy() * 100.0
    acc_s = best["val.accuracy_std"].to_numpy() * 100.0
    ep_m = best["epochs_trained_mean"].to_numpy()
    ep_s = best["epochs_trained_std"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)

    for i in range(len(best)):
        ax1.errorbar(
            acc_m[i], y[i], xerr=acc_s[i], fmt="o", color=colors[i], markersize=9, capsize=4, lw=2
        )
        ax1.annotate(
            f"{acc_m[i]:.2f} ± {acc_s[i]:.2f}",
            (acc_m[i], y[i]),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            fontsize=11,
        )
    ax1.margins(x=0.25, y=0.14)
    ax1.set_yticks(y)
    ax1.set_yticklabels([_optimizer_tick(row) for _, row in best.iterrows()], fontsize=11)
    ax1.set_xlabel("Exactitud de validación (%)")
    ax1.set_title("Exactitud de Validación")
    ax1.grid(axis="y", visible=False)

    ax2.barh(y, ep_m, xerr=ep_s, height=0.45, color=colors, ecolor="#52514e", capsize=4)
    for i, (_, row) in enumerate(best.iterrows()):
        text = f"{ep_m[i]:.0f} ± {ep_s[i]:.0f}"
        if row["n_stopped"] < row["n"]:
            text += f" (sin corte en {row['n'] - row['n_stopped']:.0f} de {row['n']:.0f})"
        ax2.annotate(
            text,
            (ep_m[i] + ep_s[i], y[i]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=11,
        )
    ax2.margins(x=0.22)
    ax2.set_xlabel("Épocas entrenadas")
    ax2.set_title("Épocas hasta Early Stopping")
    ax2.grid(axis="y", visible=False)

    fig.suptitle("Mejor η de Cada Mecanismo de Optimización", fontsize=16, weight="bold")
    fig.text(
        0.5,
        0.005,
        f"Red [784, 64, 10] · tanh + sigmoide · MSE · lote 32 · early stopping con paciencia 20 · "
        f"media ± desvío entre {best['n'].min():.0f} semillas",
        ha="center",
        fontsize=10,
        style="italic",
        color="#52514e",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return save_figure(fig, out_dir / "E2-02b_optimizers")


def plot_architecture(results_dir: Path, out_dir: Path) -> Path:
    """Exactitud de validación vs cantidad de parámetros (ancho y profundidad)."""
    apply_style()
    df_w = pd.read_csv(results_dir / "ej2_arch_width" / "summary.csv").sort_values("n_params_mean")
    df_d = pd.read_csv(results_dir / "ej2_arch_depth" / "summary.csv").sort_values("n_params_mean")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.errorbar(
        df_w["n_params_mean"] / 1000.0,
        df_w["val.accuracy_mean"] * 100.0,
        yerr=df_w["val.accuracy_std"] * 100.0,
        label="Variación de Ancho (1 capa: 16–256)",
        color=PALETTE[0],
        marker="o",
        lw=1.8,
        capsize=3,
    )

    ax.errorbar(
        df_d["n_params_mean"] / 1000.0,
        df_d["val.accuracy_mean"] * 100.0,
        yerr=df_d["val.accuracy_std"] * 100.0,
        label="Variación de Profundidad (1 a 4 capas)",
        color=PALETTE[1],
        marker="s",
        lw=1.8,
        capsize=3,
    )

    for _, r in df_w.iterrows():
        layers = json.loads(r["model.layers"])
        ax.annotate(
            f"{layers[1]}",
            (r["n_params_mean"] / 1000.0, r["val.accuracy_mean"] * 100.0),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )

    ax.set_xlabel("Parámetros entrenables (miles)")
    ax.set_ylabel("Exactitud de validación (%)")
    ax.set_title("Efecto de la Arquitectura: Ancho vs Profundidad")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)

    fig.tight_layout()
    return save_figure(fig, out_dir / "E2-03_architecture")


def plot_augmentation_impact(results_dir: Path, out_dir: Path) -> Path:
    """Impacto de data augmentation en la exactitud global y recall del dígito 5."""
    apply_style()
    df = pd.read_csv(results_dir / "ej2_extra_augmentation" / "summary.csv")

    labels = []
    accs = []
    accs_err = []
    rec5 = []
    rec5_err = []

    order = [
        ("None", 0, "Base (sin aug)"),
        ("gaussian_noise", 0.05, "Ruido σ=0.05"),
        ("random_shift", 1.0, "Shift ±1 px"),
        ("random_shift", 2.0, "Shift ±2 px"),
    ]

    for kind, param, name in order:
        if kind == "None":
            sub = df[df["training.augmentation.kind"].isna()]
        elif kind == "gaussian_noise":
            sub = df[df["training.augmentation.kind"] == "gaussian_noise"]
        else:
            sub = df[
                (df["training.augmentation.kind"] == "random_shift")
                & (df["training.augmentation.max_px"] == param)
            ]
        labels.append(name)
        accs.append(sub["val.accuracy_mean"].values[0] * 100.0)
        accs_err.append(sub["val.accuracy_std"].values[0] * 100.0)
        rec5.append(sub["val.per_class.recall.5_mean"].values[0] * 100.0)
        rec5_err.append(sub["val.per_class.recall.5_std"].values[0] * 100.0)

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        x - w / 2,
        accs,
        w,
        yerr=accs_err,
        label="Exactitud Validación Global",
        color=PALETTE[0],
        capsize=3,
    )
    ax.bar(
        x + w / 2,
        rec5,
        w,
        yerr=rec5_err,
        label="Recall Dígito 5 (Clase Minoritaria)",
        color=PALETTE[1],
        capsize=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(70, 100)
    ax.set_ylabel("Porcentaje (%)")
    ax.set_title("Efecto de Data Augmentation en Rendimiento Global y Clase Crítica")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(frameon=True, loc="lower right")

    for i in range(len(labels)):
        ax.text(
            x[i] - w / 2,
            accs[i] + 0.8,
            f"{accs[i]:.2f}%",
            ha="center",
            fontsize=8,
            weight="semibold",
        )
        ax.text(
            x[i] + w / 2,
            rec5[i] + 0.8,
            f"{rec5[i]:.1f}%",
            ha="center",
            fontsize=8,
            weight="semibold",
        )

    fig.tight_layout()
    return save_figure(fig, out_dir / "E2-04_augmentation_impact")


def plot_test_confusion_matrix(results_dir: Path, out_dir: Path) -> Path:
    """Matriz de confusión en digits_test.csv para Ej2 Final."""
    apply_style()
    with open(results_dir / "ej2_final" / "final_eval.json") as f:
        report = json.load(f)

    cm = np.array(report["models"][0]["test"]["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Muestras")

    n_classes = cm.shape[0]
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xlabel("Predicción del Modelo")
    ax.set_ylabel("Etiqueta Verdadera")
    ax.set_title(
        f"Matriz de Confusión en Test (Ej2 Final, Acc={report['mean']['accuracy'] * 100:.2f}%)"
    )

    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            color = "white" if val > thresh else "#0b0b0b"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)

    # Resaltar la fila 8 (ausente en train)
    rect = plt.Rectangle((-0.5, 7.5), 10, 1, fill=False, edgecolor="#e34948", lw=2)
    ax.add_patch(rect)
    ax.annotate(
        "Dígito 8 nunca visto\nen train (Recall = 0%)",
        xy=(9.5, 8),
        xytext=(11.5, 8),
        arrowprops=dict(arrowstyle="->", color="#e34948", lw=1.5),
        color="#e34948",
        fontsize=8,
        weight="bold",
        va="center",
    )

    fig.tight_layout()
    return save_figure(fig, out_dir / "E2-05_test_confusion_matrix")


def plot_misclassified_test(results_dir: Path, test_path: Path, out_dir: Path) -> Path:
    """Grilla de ejemplos del conjunto de test mal clasificados."""
    apply_style()
    with np.load(results_dir / "ej2_final" / "final_s0" / "test_predictions.npz") as preds:
        y_true = preds["labels_true"]
        y_pred = preds["labels_pred"]

    dataset = load_digits_csv(test_path)
    X_test = dataset.X

    wrong_idx = np.flatnonzero(y_true != y_pred)

    # Elegir 10 ejemplos: 5 del dígito 8 y 5 de otros dígitos
    wrong_8 = [i for i in wrong_idx if y_true[i] == 8][:5]
    wrong_other = [i for i in wrong_idx if y_true[i] != 8][:5]
    selected = wrong_8 + wrong_other

    fig, axes = plt.subplots(2, 5, figsize=(9, 4.2))
    for idx, ax in zip(selected, axes.ravel(), strict=True):
        img = X_test[idx].reshape(28, 28)
        ax.imshow(img, cmap="gray_r", interpolation="nearest")
        true_l = y_true[idx]
        pred_l = y_pred[idx]
        title_color = "#e34948" if true_l == 8 else "#eb6834"
        ax.set_title(
            f"Real: {true_l} → Pred: {pred_l}", fontsize=8, color=title_color, weight="bold"
        )
        ax.axis("off")

    fig.suptitle("Ejemplos de digits_test.csv Mal Clasificados por Ej2", fontsize=10, weight="bold")
    fig.tight_layout()
    return save_figure(fig, out_dir / "E2-06_misclassified_test")


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera figuras cuantitativas para Ej2")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--test-path", type=Path, default=Path("datasets/digits_test.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures/ej2"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generando figuras para Ej2 en {args.out_dir}...")

    plot_protocol(args.out_dir)
    print("✓ E2-01_evaluation_protocol")

    plot_lr_optimizer(args.results_dir, args.out_dir)
    print("✓ E2-02_lr_optimizer")

    plot_optimizers(args.results_dir, args.out_dir)
    print("✓ E2-02b_optimizers")

    plot_architecture(args.results_dir, args.out_dir)
    print("✓ E2-03_architecture")

    plot_augmentation_impact(args.results_dir, args.out_dir)
    print("✓ E2-04_augmentation_impact")

    plot_test_confusion_matrix(args.results_dir, args.out_dir)
    print("✓ E2-05_test_confusion_matrix")

    plot_misclassified_test(args.results_dir, args.test_path, args.out_dir)
    print("✓ E2-06_misclassified_test")

    print("Todas las figuras de Ej2 generadas exitosamente.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
