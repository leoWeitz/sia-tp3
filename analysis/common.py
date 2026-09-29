"""Lectura de results/, agregación por semillas y estilo común de los gráficos.

analysis/ solo lee lo que escribió experiments/ (CLAUDE.md §2.6): ningún
script de análisis entrena. Todo lo que se presenta sale de acá (Constitución
C6) y se reporta como media ± desvío entre semillas (C4).
"""

import json
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import pandas as pd

# Paleta categórica fija (validada para daltonismo y sobre fondo blanco): el
# color i es siempre el de la serie i, en todo el TP. Aqua, amarillo y magenta
# tienen contraste < 3:1 con el fondo: los gráficos que los usan llevan leyenda
# o etiquetas directas, nunca solo el color. Más de 8 series: agrupar u otro gráfico.
PALETTE = (
    "#2a78d6",  # azul
    "#eb6834",  # naranja
    "#1baf7a",  # aqua
    "#eda100",  # amarillo
    "#e87ba4",  # magenta
    "#008300",  # verde
    "#4a3aa7",  # violeta
    "#e34948",  # rojo
)
_INK = "#0b0b0b"  # texto principal
_INK_SECONDARY = "#52514e"  # rótulos
_MUTED = "#898781"  # ticks
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
FIGURE_DPI = 200

# Números por corrida de metrics.json que se agregan además de train.* y val.*.
RUN_NUMBERS = ("best_epoch", "epochs_trained", "time_total_s", "s_per_epoch", "n_params")
# Identificación de la corrida: no se agregan como métricas.
META_COLUMNS = ("seed", "fold")


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def flatten(data: Mapping[str, Any], expand_lists: bool = False, prefix: str = "") -> dict:
    """Dict anidado → claves en notación punto ({"a": {"b": 1}} → {"a.b": 1}).

    Las listas quedan como texto JSON (así sirven para agrupar, p. ej.
    model.layers), salvo con expand_lists, donde una lista de números se abre
    en una columna por elemento (val.per_class.recall.3).
    """
    out: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(flatten(value, expand_lists, prefix=f"{name}."))
        elif isinstance(value, list):
            numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
            if expand_lists and value and numbers:
                out.update({f"{name}.{i}": v for i, v in enumerate(value)})
            else:
                out[name] = json.dumps(value)
        elif _is_scalar(value):
            out[name] = value
        else:
            out[name] = json.dumps(value)
    return out


def _completed_runs(results_dir: str | PathLike) -> list[Path]:
    """Carpetas de corridas terminadas (con metrics.json) de results/<run_name>."""
    return sorted(p.parent for p in Path(results_dir).glob("*/metrics.json"))


def load_runs(results_dir: str | PathLike) -> pd.DataFrame:
    """Una fila por corrida terminada de results_dir (results/<run_name>).

    Columnas: run (carpeta), la config aplanada (training.optimizer.lr,
    model.layers como texto, ...), sweep_keys (JSON con las claves barridas) y
    metrics.json aplanado (status, best_epoch, train.loss, val.accuracy,
    val.per_class.recall.<clase>, ...).
    """
    rows = []
    for run_dir in _completed_runs(results_dir):
        config_path = run_dir / "config.json"
        config = _read_json(config_path) if config_path.exists() else {}
        row: dict[str, Any] = {"run": run_dir.name}
        row.update(flatten({k: v for k, v in config.items() if k != "sweep"}))
        row["sweep_keys"] = json.dumps(list(config.get("sweep", {})))
        row.update(flatten(_read_json(run_dir / "metrics.json"), expand_lists=True))
        rows.append(row)
    return pd.DataFrame(rows)


def load_histories(results_dir: str | PathLike) -> pd.DataFrame:
    """Historiales de las corridas terminadas en formato largo.

    Columnas: run, hash, seed, fold, epoch, metric (train_loss, val_accuracy,
    lr, elapsed_s, ...), value. Las columnas vacías del history.csv (p. ej.
    val_loss sin validación) no aparecen.
    """
    columns = ["run", "hash", "seed", "fold", "epoch", "metric", "value"]
    frames = []
    for run_dir in _completed_runs(results_dir):
        history_path = run_dir / "history.csv"
        if not history_path.exists():
            continue
        meta = _read_json(run_dir / "metrics.json")
        long = pd.read_csv(history_path).melt(
            id_vars="epoch", var_name="metric", value_name="value"
        )
        long = long.dropna(subset=["value"])
        long.insert(0, "run", run_dir.name)
        long.insert(1, "hash", meta.get("hash"))
        long.insert(2, "seed", meta.get("seed"))
        long.insert(3, "fold", meta.get("fold"))
        frames.append(long)
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


def aggregate(
    df: pd.DataFrame, by: str | Sequence[str], metrics: Sequence[str] | None = None
) -> pd.DataFrame:
    """Media, desvío, mínimo, máximo y n de cada métrica por grupo (p. ej. por hash).

    by: columna(s) de agrupación. metrics: columnas a agregar; None = todas
    las numéricas salvo by, seed y fold. El desvío es el muestral (ddof = 1),
    NaN con una sola corrida. Columnas: by..., n, <m>_mean, <m>_std, <m>_min, <m>_max.
    """
    by = [by] if isinstance(by, str) else list(by)
    if metrics is None:
        numeric = df.select_dtypes(include="number").columns
        metrics = [c for c in numeric if c not in by and c not in META_COLUMNS]
    groups = df.groupby(by, dropna=False, sort=True)
    stats = {"n": groups.size()}
    for m in metrics:
        values = groups[m]
        stats[f"{m}_mean"] = values.mean()
        stats[f"{m}_std"] = values.std(ddof=1)
        stats[f"{m}_min"] = values.min()
        stats[f"{m}_max"] = values.max()
    return pd.DataFrame(stats).reset_index()


def apply_style() -> None:
    """Estilo común de todas las figuras: legible en proyector, paleta fija, ejes discretos."""
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams.update(
        {
            "figure.figsize": (8.0, 5.0),
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "text.color": _INK,
            "axes.labelcolor": _INK_SECONDARY,
            "axes.edgecolor": _AXIS,
            "xtick.color": _MUTED,
            "ytick.color": _MUTED,
            "xtick.labelcolor": _INK_SECONDARY,
            "ytick.labelcolor": _INK_SECONDARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": _GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(color=list(PALETTE)),
            "lines.linewidth": 2.0,
            "lines.markersize": 7,
            "legend.frameon": False,
        }
    )


def save_figure(fig: Any, path: str | PathLike) -> list[Path]:
    """Guarda fig como <path>.png (200 dpi) y <path>.pdf, crea la carpeta y la cierra.

    path va sin extensión, p. ej. figures/ej2/E2-b1_lr_curves.
    """
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    paths = [path.with_name(path.name + ext) for ext in (".png", ".pdf")]
    for p in paths:
        fig.savefig(p, dpi=FIGURE_DPI)
    plt.close(fig)
    return paths
