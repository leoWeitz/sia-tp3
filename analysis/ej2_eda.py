"""Exploración de datasets de dígitos (F12 - Parte 1: EDA).

Genera figuras limpias, tablas y resumen cuantitativo para docs/datos/digits.md:
- Balance de clases comparativo entre train, test y more_digits (marcando la falta del 8 y escasez del 5).
- Dígito promedio por clase (28×28).
- Grilla de ejemplos reales por clase.
- Distribución de intensidad de píxeles y mapa de varianza (píxeles muertos en bordes).
- Análisis de duplicados intra-dataset e inter-dataset.

Uso:
    python -m analysis.ej2_eda [--train datasets/digits.csv] [--test datasets/digits_test.csv]
                               [--more datasets/more_digits.csv] [--out-dir figures/ej2/eda]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.common import PALETTE, apply_style, save_figure
from data.loaders import DIGITS_IMAGE_SHAPE, file_sha256, load_digits_csv

FIGURES = (
    "EDA-01_class_balance",
    "EDA-02_average_digits",
    "EDA-03_sample_grid",
    "EDA-04_pixel_variance_heatmaps",
)


def class_balance_table(
    y_train: np.ndarray,
    y_test: np.ndarray | None = None,
    y_more: np.ndarray | None = None,
    n_classes: int = 10,
) -> pd.DataFrame:
    """Tabla de frecuencias absolutas y relativas por clase para cada split."""
    counts_tr = np.bincount(y_train, minlength=n_classes)
    pct_tr = counts_tr / len(y_train) * 100.0 if len(y_train) > 0 else np.zeros(n_classes)

    data: dict[str, Any] = {
        "clase": list(range(n_classes)),
        "train_n": counts_tr,
        "train_pct": np.round(pct_tr, 2),
    }

    if y_test is not None:
        counts_te = np.bincount(y_test, minlength=n_classes)
        pct_te = counts_te / len(y_test) * 100.0 if len(y_test) > 0 else np.zeros(n_classes)
        data["test_n"] = counts_te
        data["test_pct"] = np.round(pct_te, 2)

    if y_more is not None:
        counts_mo = np.bincount(y_more, minlength=n_classes)
        pct_mo = counts_mo / len(y_more) * 100.0 if len(y_more) > 0 else np.zeros(n_classes)
        data["more_n"] = counts_mo
        data["more_pct"] = np.round(pct_mo, 2)

    df = pd.DataFrame(data).set_index("clase")
    return df


def compute_average_digits(
    X: np.ndarray, y: np.ndarray, n_classes: int = 10
) -> list[np.ndarray | None]:
    """Calcula la imagen promedio (28, 28) por clase. Si no hay muestras, devuelve None."""
    averages: list[np.ndarray | None] = []
    for c in range(n_classes):
        mask = y == c
        if np.any(mask):
            mean_img = X[mask].mean(axis=0).reshape(DIGITS_IMAGE_SHAPE)
            averages.append(mean_img)
        else:
            averages.append(None)
    return averages


def pixel_stats(X: np.ndarray) -> dict[str, Any]:
    """Estadísticos globales sobre los 784 píxeles."""
    variances = X.var(axis=0)
    means = X.mean(axis=0)
    zero_var = int(np.sum(variances == 0.0))
    return {
        "min": float(X.min()),
        "max": float(X.max()),
        "mean_global": float(X.mean()),
        "std_global": float(X.std()),
        "zero_var_pixels": zero_var,
        "active_pixels": int(len(variances) - zero_var),
        "zero_var_pct": float(np.round(zero_var / len(variances) * 100.0, 2)),
        "mean_map": means.reshape(DIGITS_IMAGE_SHAPE).tolist(),
        "var_map": variances.reshape(DIGITS_IMAGE_SHAPE).tolist(),
    }


def check_duplicates(datasets: dict[str, np.ndarray]) -> dict[str, Any]:
    """Verifica duplicados intra-dataset e inter-dataset basados en bytes exactos de filas."""
    hashes: dict[str, list[bytes]] = {
        name: [row.tobytes() for row in X] for name, X in datasets.items()
    }
    intra: dict[str, int] = {}
    for name, h_list in hashes.items():
        intra[name] = len(h_list) - len(set(h_list))

    names = list(datasets.keys())
    inter: dict[str, int] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            s1 = set(hashes[n1])
            s2 = set(hashes[n2])
            inter[f"{n1}_vs_{n2}"] = len(s1.intersection(s2))

    return {"intra_dataset": intra, "inter_dataset": inter}


def plot_class_balance(balance_df: pd.DataFrame, out_path: Path) -> None:
    """Genera EDA-01: gráfico de barras con el balance de clases en train, test y more_digits."""
    apply_style()
    n_classes = len(balance_df)
    x = np.arange(n_classes)

    has_test = "test_n" in balance_df.columns
    has_more = "more_n" in balance_df.columns

    n_series = 1 + int(has_test) + int(has_more)
    width = 0.8 / n_series

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Frecuencias absolutas
    offsets = np.linspace(-width * (n_series - 1) / 2, width * (n_series - 1) / 2, n_series)
    series_idx = 0

    ax1.bar(
        x + offsets[series_idx],
        balance_df["train_n"],
        width=width,
        label="digits.csv (train)",
        color=PALETTE[0],
    )
    series_idx += 1

    if has_test:
        ax1.bar(
            x + offsets[series_idx],
            balance_df["test_n"],
            width=width,
            label="digits_test.csv",
            color=PALETTE[1],
        )
        series_idx += 1

    if has_more:
        ax1.bar(
            x + offsets[series_idx],
            balance_df["more_n"],
            width=width,
            label="more_digits.csv",
            color=PALETTE[2],
        )

    ax1.set_xlabel("Dígito (clase)")
    ax1.set_ylabel("Cantidad de muestras")
    ax1.set_title("Cantidad por clase")
    ax1.set_xticks(x)
    ax1.legend(loc="upper right", frameon=True)

    # Anotación destacando el 8 ausente y el 5 escaso
    train_8 = balance_df.loc[8, "train_n"]
    train_5 = balance_df.loc[5, "train_n"]
    ax1.annotate(
        "0 muestras\n(clase 8 ausente)",
        xy=(8 + offsets[0], train_8),
        xytext=(7.2, 500),
        arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.2),
        fontsize=11,
        fontweight="bold",
    )
    ax1.annotate(
        f"{train_5} muestras\n(desbalance)",
        xy=(5 + offsets[0], train_5),
        xytext=(4.3, 750),
        arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.2),
        fontsize=11,
    )

    # Panel 2: Proporciones porcentuales (%)
    series_idx = 0
    ax2.bar(
        x + offsets[series_idx],
        balance_df["train_pct"],
        width=width,
        label="digits.csv (train)",
        color=PALETTE[0],
    )
    series_idx += 1

    if has_test:
        ax2.bar(
            x + offsets[series_idx],
            balance_df["test_pct"],
            width=width,
            label="digits_test.csv",
            color=PALETTE[1],
        )
        series_idx += 1

    if has_more:
        ax2.bar(
            x + offsets[series_idx],
            balance_df["more_pct"],
            width=width,
            label="more_digits.csv",
            color=PALETTE[2],
        )

    ax2.axhline(10.0, color="#898781", linestyle="--", linewidth=1.2, label="Balance ideal (10 %)")
    ax2.set_xlabel("Dígito (clase)")
    ax2.set_ylabel("Proporción (%)")
    ax2.set_title("Proporción porcentual")
    ax2.set_xticks(x)
    ax2.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    save_figure(fig, out_path)


def plot_average_digits(
    avg_train: list[np.ndarray | None],
    avg_test: list[np.ndarray | None] | None,
    out_path: Path,
) -> None:
    """Genera EDA-02: Dígito promedio por clase (28×28)."""
    apply_style()
    n_rows = 2 if avg_test is not None else 1
    fig, axes = plt.subplots(n_rows, 10, figsize=(15, 3.2 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    for c in range(10):
        # Fila 0: train
        ax = axes[0, c]
        img = avg_train[c]
        if img is not None:
            ax.imshow(img, cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"Train: {c}", fontsize=12)
        else:
            ax.text(
                0.5,
                0.5,
                "Sin\nmuestras\n(N = 0)",
                ha="center",
                va="center",
                color="red",
                fontsize=11,
                fontweight="bold",
                transform=ax.transAxes,
            )
            ax.set_facecolor("#f5f5f5")
            ax.set_title(f"Train: {c}", fontsize=12, color="red")
        ax.set_xticks([])
        ax.set_yticks([])

        # Fila 1: test (si existe)
        if avg_test is not None:
            ax_t = axes[1, c]
            img_t = avg_test[c]
            if img_t is not None:
                ax_t.imshow(img_t, cmap="viridis", vmin=0, vmax=1)
                ax_t.set_title(f"Test: {c}", fontsize=12)
            else:
                ax_t.text(0.5, 0.5, "N = 0", ha="center", va="center", transform=ax_t.transAxes)
            ax_t.set_xticks([])
            ax_t.set_yticks([])

    fig.suptitle("Dígito promedio por clase (intensidad media píxel a píxel)", y=1.02, fontsize=15)
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_sample_grid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray | None,
    y_test: np.ndarray | None,
    out_path: Path,
    seed: int = 42,
    n_samples: int = 5,
) -> None:
    """Genera EDA-03: Grilla de 10 clases × 5 ejemplos aleatorios reales."""
    apply_style()
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(10, n_samples, figsize=(n_samples * 1.5, 15))

    for c in range(10):
        train_indices = np.where(y_train == c)[0]
        if len(train_indices) >= n_samples:
            chosen = rng.choice(train_indices, size=n_samples, replace=False)
            source = X_train
            note = ""
        elif len(train_indices) > 0:
            chosen = rng.choice(train_indices, size=n_samples, replace=True)
            source = X_train
            note = ""
        else:
            # Caso clase ausente en train: tomar de test si está disponible
            if X_test is not None and y_test is not None:
                test_indices = np.where(y_test == c)[0]
                if len(test_indices) >= n_samples:
                    chosen = rng.choice(test_indices, size=n_samples, replace=False)
                    source = X_test
                    note = " (test)"
                elif len(test_indices) > 0:
                    chosen = rng.choice(test_indices, size=n_samples, replace=True)
                    source = X_test
                    note = " (test)"
                else:
                    chosen = []
                    source = None
                    note = " (sin datos)"
            else:
                chosen = []
                source = None
                note = " (sin datos)"

        for s in range(n_samples):
            ax = axes[c, s]
            if len(chosen) > 0 and source is not None:
                img = source[chosen[s]].reshape(DIGITS_IMAGE_SHAPE)
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_facecolor("#eaeaea")

            if s == 0:
                ax.set_ylabel(f"Dígito {c}{note}", fontsize=12, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"Muestras reales por clase ({n_samples} ejemplos aleatorios)", y=1.01, fontsize=15
    )
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_pixel_variance_heatmaps(
    X: np.ndarray, out_path: Path, title_prefix: str = "digits.csv"
) -> None:
    """Genera EDA-04: Media, varianza de píxeles e histograma de valores de entrada."""
    apply_style()
    means = X.mean(axis=0).reshape(DIGITS_IMAGE_SHAPE)
    variances = X.var(axis=0).reshape(DIGITS_IMAGE_SHAPE)
    zero_var_mask = (variances == 0.0).reshape(DIGITS_IMAGE_SHAPE)

    fig = plt.figure(figsize=(15, 4.5))

    # Panel 1: Media por píxel
    ax1 = fig.add_subplot(1, 3, 1)
    im1 = ax1.imshow(means, cmap="magma", vmin=0, vmax=means.max())
    ax1.set_title(f"Media por píxel ({title_prefix})")
    ax1.set_xticks([])
    ax1.set_yticks([])
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Panel 2: Varianza por píxel y contorno de varianza nula
    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(variances, cmap="viridis", vmin=0, vmax=variances.max())
    # Marcar píxeles muertos (bordes)
    ax2.contour(zero_var_mask, levels=[0.5], colors="red", linewidths=1.2)
    ax2.set_title("Varianza (borde rojo: var = 0)")
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Panel 3: Distribución global de intensidades
    ax3 = fig.add_subplot(1, 3, 3)
    flat_sample = X.flatten()
    # Muestra representativa de 100k píxeles para no saturar memoria/tiempo
    if len(flat_sample) > 100_000:
        rng = np.random.default_rng(0)
        flat_sample = rng.choice(flat_sample, size=100_000, replace=False)

    ax3.hist(flat_sample, bins=50, color=PALETTE[0], edgecolor="white", density=True)
    ax3.set_yscale("log")
    ax3.set_xlabel("Intensidad del píxel")
    ax3.set_ylabel("Densidad (escala log)")
    ax3.set_title("Distribución de valores de píxeles")

    fig.tight_layout()
    save_figure(fig, out_path)


def run_eda(
    train_path: Path,
    test_path: Path | None,
    more_path: Path | None,
    out_dir: Path,
) -> dict[str, Any]:
    """Ejecuta la exploración completa de los datasets de dígitos."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar datasets
    ds_train = load_digits_csv(train_path)
    X_train, y_train = ds_train.X, ds_train.y

    X_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    test_sha = None
    if test_path is not None and test_path.exists():
        ds_test = load_digits_csv(test_path)
        X_test, y_test = ds_test.X, ds_test.y
        test_sha = ds_test.meta.get("sha256")

    X_more: np.ndarray | None = None
    y_more: np.ndarray | None = None
    more_sha = None
    if more_path is not None and more_path.exists():
        ds_more = load_digits_csv(more_path)
        X_more, y_more = ds_more.X, ds_more.y
        more_sha = ds_more.meta.get("sha256")

    # 2. Balance de clases
    df_balance = class_balance_table(y_train, y_test, y_more)
    df_balance.to_csv(out_dir / "class_balance.csv")

    # 3. Dígito promedio
    avg_train = compute_average_digits(X_train, y_train)
    avg_test = compute_average_digits(X_test, y_test) if X_test is not None and y_test is not None else None

    # 4. Estadísticos de píxeles
    stats_tr = pixel_stats(X_train)
    stats_summary = {
        "min": stats_tr["min"],
        "max": stats_tr["max"],
        "mean_global": stats_tr["mean_global"],
        "std_global": stats_tr["std_global"],
        "zero_var_pixels": stats_tr["zero_var_pixels"],
        "active_pixels": stats_tr["active_pixels"],
        "zero_var_pct": stats_tr["zero_var_pct"],
    }
    pd.DataFrame([stats_summary]).to_csv(out_dir / "pixel_stats.csv", index=False)

    # 5. Duplicados
    datasets_map = {"train": X_train}
    if X_test is not None:
        datasets_map["test"] = X_test
    if X_more is not None:
        datasets_map["more"] = X_more
    dups_info = check_duplicates(datasets_map)

    # 6. Generar figuras
    plot_class_balance(df_balance, out_dir / "EDA-01_class_balance")
    plot_average_digits(avg_train, avg_test, out_dir / "EDA-02_average_digits")
    plot_sample_grid(X_train, y_train, X_test, y_test, out_dir / "EDA-03_sample_grid")
    plot_pixel_variance_heatmaps(X_train, out_dir / "EDA-04_pixel_variance_heatmaps")

    # 7. Resumen estructurado JSON
    max_te_acc = None
    if y_test is not None:
        # Cota superior si se fallan todos los 8
        n_test = len(y_test)
        n_test_8 = int(np.sum(y_test == 8))
        max_te_acc = float(np.round((n_test - n_test_8) / n_test * 100.0, 2))

    resumen = {
        "datasets": {
            "train": {
                "path": str(train_path),
                "sha256": ds_train.meta.get("sha256") or file_sha256(train_path),
                "n_rows": int(len(y_train)),
                "n_features": int(X_train.shape[1]),
                "clase_8_presente": bool(np.any(y_train == 8)),
                "clase_8_n": int(np.sum(y_train == 8)),
                "clase_5_n": int(np.sum(y_train == 5)),
            },
            "test": {
                "path": str(test_path) if test_path else None,
                "sha256": test_sha,
                "n_rows": int(len(y_test)) if y_test is not None else None,
                "clase_8_presente": bool(np.any(y_test == 8)) if y_test is not None else None,
                "clase_8_n": int(np.sum(y_test == 8)) if y_test is not None else None,
            },
            "more": {
                "path": str(more_path) if more_path else None,
                "sha256": more_sha,
                "n_rows": int(len(y_more)) if y_more is not None else None,
            },
        },
        "class_balance": df_balance.to_dict(orient="index"),
        "pixel_stats": stats_summary,
        "duplicates": dups_info,
        "conclusions": {
            "ausencia_8_train": True if np.sum(y_train == 8) == 0 else False,
            "techo_teorico_accuracy_test": max_te_acc,
            "decision_normalizacion": "none (píxeles en [0, 1], 97 píxeles con varianza 0 en bordes)",
            "decision_target_encoding": "onehot (10 neuronas de salida)",
            "protocolo_validacion_default": "holdout 80/20 estratificado (3 semillas)",
            "protocolo_validacion_alternativo": "5-fold estratificado",
        },
    }

    with open(out_dir / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    return resumen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EDA de dígitos (F12).")
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("datasets/digits.csv"),
        help="Ruta al CSV de entrenamiento (digits.csv).",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("datasets/digits_test.csv"),
        help="Ruta al CSV de test (digits_test.csv).",
    )
    parser.add_argument(
        "--more",
        type=Path,
        default=Path("datasets/more_digits.csv"),
        help="Ruta al CSV de more_digits (opcional).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures/ej2/eda"),
        help="Directorio donde guardar figuras y reportes.",
    )
    args = parser.parse_args(argv)

    if not args.train.exists():
        print(f"Error: no existe {args.train}", file=sys.stderr)
        return 2

    run_eda(
        train_path=args.train,
        test_path=args.test if args.test.exists() else None,
        more_path=args.more if args.more.exists() else None,
        out_dir=args.out_dir,
    )
    print(f"EDA completada exitosamente. Resultados en {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
