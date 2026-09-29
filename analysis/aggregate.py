"""Resumen de un experimento: results/<run_name>/summary.csv.

    python -m analysis.aggregate results/<run_name>

Una fila por configuración (hash8) con las claves del sweep, n (corridas),
n_ok, n_diverged y, para cada métrica final (train.*, val.* y los números de
RUN_NUMBERS), media, desvío (ddof = 1), mínimo y máximo entre semillas/folds.
Solo lee: nunca entrena (CLAUDE.md §2.6).
"""

import argparse
import json
import sys
from collections.abc import Sequence
from os import PathLike
from pathlib import Path

import pandas as pd

from analysis.common import RUN_NUMBERS, aggregate, load_runs


def _metric_columns(runs: pd.DataFrame) -> list[str]:
    numeric = set(runs.select_dtypes(include="number").columns)
    split_metrics = [
        c
        for c in runs.columns
        if c.startswith(("train.", "val.")) and c in numeric and ".per_class.support." not in c
    ]
    return split_metrics + [c for c in RUN_NUMBERS if c in numeric]


def _sweep_columns(runs: pd.DataFrame) -> list[str]:
    """Columnas de la config que se barrieron (una clave puede ser una sección entera)."""
    keys: list[str] = []
    for value in runs["sweep_keys"]:
        keys += [k for k in json.loads(value) if k not in keys]
    return [c for k in keys for c in runs.columns if c == k or c.startswith(k + ".")]


def summarize(results_dir: str | PathLike) -> pd.DataFrame:
    """summary.csv de results_dir como DataFrame (sin escribirlo)."""
    runs = load_runs(results_dir)
    if runs.empty:
        raise ValueError(f"{results_dir}: no hay corridas terminadas (carpetas con metrics.json)")
    summary = aggregate(runs, "hash", _metric_columns(runs))
    groups = runs.groupby("hash", sort=True)
    sweep = _sweep_columns(runs)
    extra = pd.DataFrame(
        {
            **{c: groups[c].first() for c in sweep},
            "n_ok": groups["status"].apply(lambda s: int((s == "ok").sum())),
            "n_diverged": groups["status"].apply(lambda s: int((s == "diverged").sum())),
        }
    ).reset_index()
    summary = summary.merge(extra, on="hash")
    first = ["hash", *sweep, "n", "n_ok", "n_diverged"]
    return summary[first + [c for c in summary.columns if c not in first]]


def main(argv: Sequence[str] | None = None) -> int:
    """Escribe results/<run_name>/summary.csv. Devuelve 0, o 2 si no hay corridas."""
    parser = argparse.ArgumentParser(
        prog="python -m analysis.aggregate",
        description="Agrega las corridas de un experimento en results/<run_name>/summary.csv.",
    )
    parser.add_argument("results_dir", help="carpeta del experimento, p. ej. results/ej2_lr")
    args = parser.parse_args(argv)
    try:
        summary = summarize(args.results_dir)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    path = Path(args.results_dir) / "summary.csv"
    summary.to_csv(path, index=False)
    print(f"{path}: {len(summary)} configuración(es), {int(summary['n'].sum())} corridas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
