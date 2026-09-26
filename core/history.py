"""Registro época a época de un entrenamiento.

Cada época es un dict con las claves de COLUMNS que correspondan: val_loss y
val_metric solo aparecen si fit recibió validación, y las métricas solo si
recibió una función de métrica. Al exportar, lo que falta queda vacío.
"""

import csv
from os import PathLike

import numpy as np

# Columnas de history.csv, en este orden (ver contrato de resultados en CLAUDE.md).
COLUMNS = ("epoch", "train_loss", "val_loss", "train_metric", "val_metric", "elapsed_s")


class History:
    def __init__(self) -> None:
        self.records: list[dict[str, float]] = []

    def append(self, logs: dict[str, float]) -> None:
        unknown = set(logs) - set(COLUMNS)
        if unknown:
            raise ValueError(f"Claves desconocidas en el historial: {sorted(unknown)}")
        self.records.append(dict(logs))

    def column(self, name: str) -> np.ndarray:
        """Valores de una columna por época, con NaN donde no se registró. Forma (n_epocas,)."""
        if name not in COLUMNS:
            raise ValueError(f"Columna desconocida: {name!r}. Opciones: {list(COLUMNS)}")
        return np.array([r.get(name, np.nan) for r in self.records], dtype=float)

    def to_csv(self, path: str | PathLike) -> None:
        """Escribe una fila por época con las columnas de COLUMNS."""
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
            writer.writeheader()
            writer.writerows(self.records)

    def __len__(self) -> int:
        return len(self.records)
