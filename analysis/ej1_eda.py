"""Exploración de fraud_dataset.csv (F09): figuras limpias y números del diccionario de datos.

    python -m analysis.ej1_eda [--data datasets/fraud_dataset.csv] [--out-dir figures/ej1/eda]

Solo lee el CSV y grafica: no entrena nada ni modifica el archivo. Los números
de docs/datos/fraud_dataset.md salen de <out-dir>/resumen.json y
<out-dir>/rangos.csv, que escribe este script.

Figuras (FIGURES), en <out-dir> como .png y .pdf:
- EDA-01_class_balance: cantidad y proporción de flagged_fraud.
- EDA-02_target_by_class: histograma de big_model_fraud_probability por clase,
  con el corte que separa las clases.
- EDA-03_feature_histograms: densidad de cada feature por clase.
- EDA-04_feature_boxplots: boxplots de cada feature por clase.
- EDA-05_correlation: correlación de Pearson entre todas las columnas.
- EDA-06_feature_vs_target: probabilidad media de BigModel y tasa de fraude
  por decil de cada feature (relación con el target, linealidad).
- EDA-07_bigmodel_roc_pr: ROC y PR de BigModel contra flagged_fraud (la
  performance de referencia del TinyModel).
- EDA-08_ranges_table: tabla de rangos y tipos.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.common import PALETTE, apply_style, save_figure
from core.thresholds import auc_trapezoid, average_precision, pr_curve, roc_curve
from data.loaders import file_sha256
from data.splits import stratify_labels

TARGET = "big_model_fraud_probability"
LABEL = "flagged_fraud"
# Columnas del CSV, en el orden del archivo.
COLUMNS = (
    "timestamp",
    "amount_usd",
    "quantity_purchased",
    "session_duration_seconds",
    "days_since_last_purchase",
    "account_age_days",
    "device_screen_resolution",
    "time_since_last_login_s",
    "items_viewed_before_purchase",
    TARGET,
    LABEL,
)
RAW_FEATURES = COLUMNS[:-2]
# Features con cola pesada: se grafican en escala log.
LOG_SCALE = ("amount_usd", "time_since_last_login_s")
# Resoluciones de pantalla comunes (ancho × alto → píxeles totales).
COMMON_RESOLUTIONS = {
    f"{w}x{h}": w * h
    for w, h in (
        (1280, 720),
        (1280, 800),
        (1366, 768),
        (1440, 900),
        (1536, 864),
        (1600, 900),
        (1920, 1080),
        (1920, 1200),
        (2560, 1440),
        (2560, 1600),
        (3840, 2160),
    )
}
CLASS_NAMES = ("no fraude", "fraude")
CLASS_COLORS = (PALETTE[0], PALETTE[1])
FIGURES = (
    "EDA-01_class_balance",
    "EDA-02_target_by_class",
    "EDA-03_feature_histograms",
    "EDA-04_feature_boxplots",
    "EDA-05_correlation",
    "EDA-06_feature_vs_target",
    "EDA-07_bigmodel_roc_pr",
    "EDA-08_ranges_table",
)


def read_fraud(path: str | Path) -> pd.DataFrame:
    """Lee el CSV y verifica que tenga exactamente las columnas documentadas (COLUMNS)."""
    df = pd.read_csv(path)
    if tuple(df.columns) != COLUMNS:
        raise ValueError(f"{path}: columnas {list(df.columns)}, se esperaban {list(COLUMNS)}")
    return df


def column_kind(values: pd.Series) -> str:
    """'binaria' (solo 0/1), 'entera' (dtype entero) o 'continua'."""
    if set(values.unique()) <= {0, 1} and values.nunique() == 2:
        return "binaria"
    if pd.api.types.is_integer_dtype(values):
        return "entera"
    return "continua"


def ranges_table(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por columna: tipo, min, p1, p50, p99, max, media, desvío, asimetría y únicos."""
    rows = {}
    for col in df.columns:
        v = df[col]
        p1, p50, p99 = np.percentile(v.to_numpy(dtype=np.float64), [1, 50, 99])
        rows[col] = {
            "tipo": column_kind(v),
            "min": v.min(),
            "p1": p1,
            "p50": p50,
            "p99": p99,
            "max": v.max(),
            "media": v.mean(),
            "desvio": v.std(),
            "asimetria": v.skew(),
            "unicos": v.nunique(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def separation(df: pd.DataFrame) -> dict[str, Any]:
    """Qué tan bien separa TARGET a LABEL: máx. de los negativos, mín. de los positivos.

    separable = todo positivo tiene probabilidad mayor que todo negativo;
    n_solapados = muestras en el intervalo donde se mezclan las clases.
    """
    p, y = df[TARGET].to_numpy(), df[LABEL].to_numpy()
    max_neg, min_pos = float(p[y == 0].max()), float(p[y == 1].min())
    lo, hi = min(max_neg, min_pos), max(max_neg, min_pos)
    overlap = int(((p >= lo) & (p <= hi)).sum()) if max_neg >= min_pos else 0
    return {
        "max_negativo": max_neg,
        "min_positivo": min_pos,
        "separable": max_neg < min_pos,
        "corte_medio": (max_neg + min_pos) / 2,
        "n_solapados": overlap,
    }


def class_bounds(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Mín. y máx. de cada feature por clase y positivos fuera del rango de los negativos.

    Un valor fuera de [min, max] de los negativos solo aparece en fraudes: es
    una regla dura del dataset (p. ej. quantity_purchased > 9 → siempre fraude).
    """
    neg, pos = df[df[LABEL] == 0], df[df[LABEL] == 1]
    out = {}
    for col in RAW_FEATURES:
        lo, hi = neg[col].min(), neg[col].max()
        out[col] = {
            "neg_min": float(lo),
            "neg_max": float(hi),
            "pos_min": float(pos[col].min()),
            "pos_max": float(pos[col].max()),
            "pos_fuera_de_rango_neg": int(((pos[col] < lo) | (pos[col] > hi)).sum()),
        }
    return out


def spikes(df: pd.DataFrame, min_count: int = 20) -> list[dict[str, Any]]:
    """Valores exactos repetidos ≥ min_count veces en features casi continuas (> 100 únicos).

    En una variable continua, un valor repetido muchas veces (p. ej. un monto
    redondo o un mínimo recortado) es un patrón que conviene mirar aparte.
    """
    found = []
    for col in RAW_FEATURES:
        if col == "timestamp" or col not in df or df[col].nunique() <= 100:
            continue
        counts = df[col].value_counts()
        for value, n in counts[counts >= min_count].items():
            mask = df[col] == value
            found.append(
                {
                    "columna": col,
                    "valor": float(value),
                    "n": int(n),
                    "tasa_positivos": float(df.loc[mask, LABEL].mean()),
                    "prob_media": float(df.loc[mask, TARGET].mean()),
                }
            )
    return found


def resolution_clusters(values: np.ndarray) -> pd.DataFrame:
    """Asigna cada valor de device_screen_resolution a la resolución común más cercana.

    values: (p,) píxeles totales. Devuelve (p filas): resolucion, nominal_px, desvio_px.
    """
    values = np.asarray(values, dtype=np.int64)
    names = np.array(list(COMMON_RESOLUTIONS))
    nominal = np.array(list(COMMON_RESOLUTIONS.values()), dtype=np.int64)
    idx = np.abs(values[:, None] - nominal[None, :]).argmin(axis=1)
    return pd.DataFrame(
        {"resolucion": names[idx], "nominal_px": nominal[idx], "desvio_px": values - nominal[idx]}
    )


def decile_profile(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Probabilidad media de BigModel y tasa de fraude por decil de col (empates por orden)."""
    decile = pd.qcut(df[col].rank(method="first"), 10, labels=False)
    grouped = df.groupby(decile)
    return pd.DataFrame(
        {
            "decil": np.arange(1, 11),
            "desde": grouped[col].min().to_numpy(),
            "hasta": grouped[col].max().to_numpy(),
            "n": grouped.size().to_numpy(),
            TARGET: grouped[TARGET].mean().to_numpy(),
            LABEL: grouped[LABEL].mean().to_numpy(),
        }
    )


def strata_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de fraude en cada estrato por cuantiles del target (los que usa el runner).

    Con target continuo, data.splits.stratify_labels define los estratos:
    esto mide si estratificar por la probabilidad mantiene la proporción de LABEL.
    """
    strata = stratify_labels(df[TARGET].to_numpy())
    grouped = df.groupby(strata)
    return pd.DataFrame(
        {
            "estrato": grouped.size().index.to_numpy(),
            "n": grouped.size().to_numpy(),
            "prob_min": grouped[TARGET].min().to_numpy(),
            "prob_max": grouped[TARGET].max().to_numpy(),
            "tasa_positivos": grouped[LABEL].mean().to_numpy(),
        }
    )


def time_profile(df: pd.DataFrame) -> dict[str, Any]:
    """Rango de fechas y dispersión de la tasa de fraude / prob. media por hora, día y mes."""
    t = pd.to_datetime(df["timestamp"], unit="s")
    out: dict[str, Any] = {"desde": str(t.min()), "hasta": str(t.max())}
    for name, key in (
        ("hora", t.dt.hour),
        ("dia_semana", t.dt.dayofweek),
        ("mes", t.dt.to_period("M")),
    ):
        g = df.groupby(key)[[TARGET, LABEL]].mean()
        out[name] = {
            "grupos": len(g),
            "tasa_positivos_min": float(g[LABEL].min()),
            "tasa_positivos_max": float(g[LABEL].max()),
            "prob_media_min": float(g[TARGET].min()),
            "prob_media_max": float(g[TARGET].max()),
        }
    return out


def bigmodel_reference(df: pd.DataFrame) -> dict[str, Any]:
    """ROC-AUC, AP y precisión/recall de BigModel umbralizado en 0.5 y en el corte medio."""
    y, p = df[LABEL].to_numpy(), df[TARGET].to_numpy()
    fpr, tpr, _ = roc_curve(y, p)
    out: dict[str, Any] = {
        "roc_auc": auc_trapezoid(fpr, tpr),
        "average_precision": average_precision(y, p),
    }
    for name, t in (("umbral_0.5", 0.5), ("umbral_corte_medio", separation(df)["corte_medio"])):
        pred = p >= t
        tp = int((pred & (y == 1)).sum())
        out[name] = {
            "umbral": float(t),
            "precision": tp / max(int(pred.sum()), 1),
            "recall": tp / max(int((y == 1).sum()), 1),
        }
    return out


def summary(df: pd.DataFrame, path: str | Path) -> dict[str, Any]:
    """Todos los números del diccionario de datos (resumen.json)."""
    features = list(RAW_FEATURES)
    res = resolution_clusters(df["device_screen_resolution"].to_numpy())
    by_res = df.groupby(res["resolucion"].to_numpy())
    return {
        "archivo": str(path),
        "sha256": file_sha256(path),
        "filas": len(df),
        "columnas": len(df.columns),
        "nan_total": int(df.isna().sum().sum()),
        "duplicados": int(df.duplicated().sum()),
        "duplicados_sin_target_ni_etiqueta": int(df[features].duplicated().sum()),
        "n_positivos": int(df[LABEL].sum()),
        "tasa_positivos": float(df[LABEL].mean()),
        "target_fuera_de_0_1": int(((df[TARGET] < 0) | (df[TARGET] > 1)).sum()),
        "negativos_en_features": {c: int((df[c] < 0).sum()) for c in features},
        "correlacion": {
            c: {
                "pearson_target": float(df[c].corr(df[TARGET])),
                "spearman_target": float(df[c].corr(df[TARGET], method="spearman")),
                "pearson_etiqueta": float(df[c].corr(df[LABEL])),
            }
            for c in features
        },
        "correlacion_target_etiqueta": float(df[TARGET].corr(df[LABEL])),
        "separacion": separation(df),
        "bigmodel": bigmodel_reference(df),
        "spikes": spikes(df),
        "cotas_por_clase": class_bounds(df),
        "resolucion": {
            "desvio_px_std": float(res["desvio_px"].std()),
            "desvio_px_max_abs": int(res["desvio_px"].abs().max()),
            "grupos": {
                name: {
                    "n": int(len(g)),
                    "tasa_positivos": float(g[LABEL].mean()),
                    "prob_media": float(g[TARGET].mean()),
                }
                for name, g in by_res
            },
        },
        "tiempo": time_profile(df),
        "estratos_por_cuantiles": strata_balance(df).to_dict(orient="records"),
        "deciles": {c: decile_profile(df, c).to_dict(orient="records") for c in features},
    }


# --- figuras ---


def _feature_grid(title: str) -> tuple[Any, np.ndarray]:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(title)
    return fig, axes.ravel()


def _grid_legend(fig: Any, handles: Sequence[Any], labels: Sequence[str]) -> None:
    """Leyenda común debajo del título de una grilla de features."""
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncols=len(labels))
    fig.tight_layout(rect=(0, 0, 1, 0.94))


def fig_class_balance(df: pd.DataFrame) -> Any:
    import matplotlib.pyplot as plt

    counts = df[LABEL].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(CLASS_NAMES, counts.to_numpy(), color=CLASS_COLORS, height=0.6)
    for i, n in enumerate(counts):
        ax.text(n, i, f"  {n} ({n / len(df):.1%})", va="center")
    ax.set_xlabel("transacciones")
    ax.set_xlim(0, counts.max() * 1.25)
    ax.set_title(f"flagged_fraud (n = {len(df)})")
    ax.grid(axis="y", visible=False)
    return fig


def fig_target_by_class(df: pd.DataFrame) -> Any:
    import matplotlib.pyplot as plt

    sep = separation(df)
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots()
    parts = [df.loc[df[LABEL] == k, TARGET] for k in (0, 1)]
    ax.hist(
        parts, bins=bins, stacked=True, color=CLASS_COLORS, label=CLASS_NAMES, edgecolor="white"
    )
    ax.axvline(sep["corte_medio"], color="#52514e", linestyle="--", linewidth=1.5)
    ax.text(
        sep["corte_medio"],
        ax.get_ylim()[1] * 0.95,
        f"corte {sep['corte_medio']:.3f} ",
        ha="right",
        va="top",
        color="#52514e",
    )
    ax.set_xlabel("big_model_fraud_probability")
    ax.set_ylabel("transacciones")
    ax.set_title("Probabilidad de BigModel por clase (apilado)")
    ax.legend()
    return fig


def fig_feature_histograms(df: pd.DataFrame) -> Any:
    fig, axes = _feature_grid("Distribución de cada feature por clase (densidad)")
    handles = []
    for ax, col in zip(axes, RAW_FEATURES, strict=True):
        v = df[col].to_numpy(dtype=np.float64)
        log = col in LOG_SCALE
        if log:
            bins = np.geomspace(v.min(), v.max(), 41)
        elif df[col].nunique() <= 50:  # enteras de pocos valores: un bin por entero
            bins = np.arange(v.min() - 0.5, v.max() + 1.5)
        else:
            bins = np.linspace(v.min(), v.max(), 41)
        handles = []
        for k in (0, 1):
            _, _, patch = ax.hist(
                v[df[LABEL] == k], bins=bins, density=True, histtype="step", color=CLASS_COLORS[k]
            )
            handles.append(patch[0])
        if log:
            ax.set_xscale("log")
        ax.set_title(col + (" (log)" if log else ""), fontsize=13)
        ax.set_yticks([])
    _grid_legend(fig, handles, CLASS_NAMES)
    return fig


def fig_feature_boxplots(df: pd.DataFrame) -> Any:
    fig, axes = _feature_grid("Features por clase")
    for ax, col in zip(axes, RAW_FEATURES, strict=True):
        data = [df.loc[df[LABEL] == k, col].to_numpy() for k in (0, 1)]
        box = ax.boxplot(
            data,
            tick_labels=CLASS_NAMES,
            patch_artist=True,
            widths=0.5,
            flierprops={"markersize": 2, "alpha": 0.4},
            medianprops={"color": "#0b0b0b"},
        )
        for patch, color in zip(box["boxes"], CLASS_COLORS, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        if col in LOG_SCALE:
            ax.set_yscale("log")
        ax.set_title(col + (" (log)" if col in LOG_SCALE else ""), fontsize=13)
    fig.tight_layout()
    return fig


def fig_correlation(df: pd.DataFrame) -> Any:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    corr = df.corr()
    cmap = LinearSegmentedColormap.from_list("div", [PALETTE[0], "#f0efea", PALETTE[7]])
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=11)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="correlación de Pearson")
    ax.set_title("Correlación entre columnas")
    return fig


def fig_feature_vs_target(df: pd.DataFrame) -> Any:
    fig, axes = _feature_grid(
        "Probabilidad media de BigModel y tasa de fraude por decil de cada feature"
    )
    handles = []
    for ax, col in zip(axes, RAW_FEATURES, strict=True):
        prof = decile_profile(df, col)
        handles = [
            ax.plot(prof["decil"], prof[TARGET], marker="o", color=PALETTE[0])[0],
            ax.plot(prof["decil"], prof[LABEL], marker="s", color=PALETTE[1])[0],
        ]
        ax.set_ylim(0, 1)
        ax.set_xticks(range(1, 11))
        ax.set_title(col, fontsize=13)
    for ax in axes[6:]:
        ax.set_xlabel("decil de la feature")
    _grid_legend(fig, handles, ["prob. media BigModel", "tasa de flagged_fraud"])
    return fig


def fig_bigmodel_roc_pr(df: pd.DataFrame) -> Any:
    import matplotlib.pyplot as plt

    y, p = df[LABEL].to_numpy(), df[TARGET].to_numpy()
    ref = bigmodel_reference(df)
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = pr_curve(y, p)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
    a1.plot(fpr, tpr, color=PALETTE[0])
    a1.plot([0, 1], [0, 1], color="#c3c2b7", linestyle="--", linewidth=1)
    a1.set_xlabel("tasa de falsos positivos")
    a1.set_ylabel("tasa de verdaderos positivos")
    a1.set_title(f"ROC · AUC = {ref['roc_auc']:.3f}")
    a2.plot(rec, prec, color=PALETTE[0])
    a2.axhline(df[LABEL].mean(), color="#c3c2b7", linestyle="--", linewidth=1)
    a2.set_xlabel("recall")
    a2.set_ylabel("precision")
    a2.set_title(f"PR · AP = {ref['average_precision']:.3f}")
    for ax in (a1, a2):
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    fig.suptitle("BigModel contra flagged_fraud (referencia del TinyModel)")
    fig.tight_layout()
    return fig


def _fmt(x: Any) -> str:
    if isinstance(x, str):
        return x
    x = float(x)
    if x != 0 and (abs(x) >= 1e5 or abs(x) < 1e-2):
        return f"{x:.3g}"
    return f"{x:,.2f}".rstrip("0").rstrip(".")


def fig_ranges_table(table: pd.DataFrame) -> Any:
    import matplotlib.pyplot as plt

    cols = ["tipo", "min", "p1", "p50", "p99", "max", "asimetria", "unicos"]
    cells = [[_fmt(v) for v in row] for row in table[cols].itertuples(index=False)]
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.axis("off")
    tab = ax.table(cellText=cells, rowLabels=list(table.index), colLabels=cols, loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(11)
    tab.scale(1, 1.6)
    for (r, _), cell in tab.get_celld().items():
        cell.set_edgecolor("#e1e0d9")
        if r == 0:
            cell.set_text_props(weight="bold")
    ax.set_title("Rangos y tipos de fraud_dataset.csv")
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    """Genera las figuras, rangos.csv y resumen.json. Devuelve 0, o 2 si no está el CSV."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        prog="python -m analysis.ej1_eda",
        description="Figuras de la EDA de fraud_dataset.csv (F09).",
    )
    parser.add_argument("--data", default="datasets/fraud_dataset.csv", help="ruta del CSV")
    parser.add_argument("--out-dir", default="figures/ej1/eda", help="carpeta de las figuras")
    args = parser.parse_args(argv)
    data, out_dir = Path(args.data), Path(args.out_dir)
    if not data.exists():
        print(f"No existe {data}: copiarlo del campus a datasets/", file=sys.stderr)
        return 2

    df = read_fraud(data)
    table = ranges_table(df)
    apply_style()
    figures = {
        "EDA-01_class_balance": fig_class_balance(df),
        "EDA-02_target_by_class": fig_target_by_class(df),
        "EDA-03_feature_histograms": fig_feature_histograms(df),
        "EDA-04_feature_boxplots": fig_feature_boxplots(df),
        "EDA-05_correlation": fig_correlation(df),
        "EDA-06_feature_vs_target": fig_feature_vs_target(df),
        "EDA-07_bigmodel_roc_pr": fig_bigmodel_roc_pr(df),
        "EDA-08_ranges_table": fig_ranges_table(table),
    }
    assert tuple(figures) == FIGURES
    for name, fig in figures.items():
        save_figure(fig, out_dir / name)

    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "rangos.csv", float_format="%.6g")
    resumen = summary(df, data)
    (out_dir / "resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False, default=float) + "\n", encoding="utf-8"
    )
    sep, ref = resumen["separacion"], resumen["bigmodel"]
    print(f"{resumen['filas']} filas · positivos {resumen['tasa_positivos']:.2%}")
    print(
        f"BigModel: ROC-AUC {ref['roc_auc']:.4f} · AP {ref['average_precision']:.4f} · "
        f"separable={sep['separable']} "
        f"(máx. neg {sep['max_negativo']:.6f}, mín. pos {sep['min_positivo']:.6f})"
    )
    print(f"Figuras en {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
