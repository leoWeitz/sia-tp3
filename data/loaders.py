"""Lectura de los CSV del TP a un Dataset con X (p, n) float64 e y crudo.

No imputa ni limpia en silencio: toda fila descartada o columna codificada
queda registrada en Dataset.meta, junto con el sha256 del archivo (el runner
lo copia a metrics.json). Normalizar y particionar es trabajo de
data.preprocess y data.splits.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

DIGITS_IMAGE_SHAPE = (28, 28)
DIGITS_N_PIXELS = DIGITS_IMAGE_SHAPE[0] * DIGITS_IMAGE_SHAPE[1]


@dataclass
class Dataset:
    """Datos crudos de un CSV.

    X: (p, n) float64 · y: (p,) o (p, k), target sin escalar ni codificar.
    meta: path, sha256, n_rows, filas descartadas, columnas codificadas, etc.
    """

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_name: str
    meta: dict[str, Any] = field(default_factory=dict)


def file_sha256(path: str | Path) -> str:
    """sha256 del contenido del archivo, en hex."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _missing(columns: Sequence[str], available: pd.Index, what: str, path: Path) -> None:
    missing = [c for c in columns if c not in available]
    if missing:
        raise ValueError(
            f"{path}: {what} inexistente(s): {', '.join(repr(c) for c in missing)}. "
            f"Columnas del archivo: {list(available)}"
        )


def load_csv(
    path: str | Path,
    target: str,
    features: list[str] | None = None,
    drop: Sequence[str] = (),
    categorical: Sequence[str] = (),
    na_policy: Literal["error", "drop_rows"] = "error",
) -> Dataset:
    """Lee un CSV tabular: una columna target y features numéricas.

    features: columnas a usar, en ese orden; None = todas salvo target y drop.
    categorical: columnas (entre las features) que se codifican one-hot en su
    lugar, con nombres "col=valor" y categorías ordenadas. NaN solo se revisa en
    las columnas usadas (features + target): con na_policy="error" es un error
    que informa la cantidad por columna; con "drop_rows" se descartan esas filas
    y se registran en meta.
    """
    if na_policy not in ("error", "drop_rows"):
        raise ValueError(f"na_policy desconocida: {na_policy!r}. Opciones: ['drop_rows', 'error']")
    path = Path(path)
    df = pd.read_csv(path)

    _missing([target], df.columns, "target", path)
    _missing(list(drop), df.columns, "columna(s) de drop", path)
    if features is None:
        features = [c for c in df.columns if c != target and c not in set(drop)]
    else:
        _missing(features, df.columns, "feature(s)", path)
    if target in features:
        raise ValueError(f"El target {target!r} no puede estar entre las features")
    _missing(
        list(categorical), pd.Index(features), "columna(s) categorical entre las features", path
    )

    used = df[[*features, target]]
    na_counts = used.isna().sum()
    na_counts = na_counts[na_counts > 0]
    dropped: list[int] = []
    if len(na_counts):
        if na_policy == "error":
            detail = ", ".join(f"{col}: {n}" for col, n in na_counts.items())
            raise ValueError(
                f"{path}: hay NaN en columnas usadas ({detail}). "
                "Usar na_policy='drop_rows' o descartar las columnas con drop"
            )
        mask = used.isna().any(axis=1).to_numpy()
        dropped = np.flatnonzero(mask).tolist()
        used = used.loc[~mask]

    parts: list[pd.DataFrame] = []
    encoded: dict[str, list[Any]] = {}
    for col in features:
        if col in categorical:
            values = sorted(used[col].unique().tolist())
            encoded[col] = values
            parts.append(
                pd.DataFrame(
                    {f"{col}={v}": (used[col] == v).astype(np.float64) for v in values},
                    index=used.index,
                )
            )
        else:
            if not pd.api.types.is_numeric_dtype(used[col]):
                raise ValueError(
                    f"{path}: la feature {col!r} no es numérica (dtype {used[col].dtype}). "
                    "Pasarla en categorical o descartarla con drop"
                )
            parts.append(used[[col]])
    X_df = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=used.index)

    y = used[target].to_numpy()
    if not np.issubdtype(y.dtype, np.integer):
        y = y.astype(np.float64)

    return Dataset(
        X=X_df.to_numpy(dtype=np.float64),
        y=y,
        feature_names=[str(c) for c in X_df.columns],
        target_name=target,
        meta={
            "path": str(path),
            "sha256": file_sha256(path),
            "n_rows": int(len(used)),
            "n_rows_file": int(len(df)),
            "dropped_rows": len(dropped),
            "dropped_row_indices": dropped,
            "encoded_columns": encoded,
        },
    )


def _parse_digits(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    _missing(["label", "image"], df.columns, "columna(s)", path)
    # Un solo json.loads para todas las filas: más rápido que uno por fila.
    images = json.loads("[" + ",".join(df["image"]) + "]")
    lengths = {len(img) for img in images}
    if lengths != {DIGITS_N_PIXELS}:
        raise ValueError(
            f"{path}: cada imagen tiene que tener {DIGITS_N_PIXELS} píxeles, "
            f"hay largos {sorted(lengths)}"
        )
    X = np.array(images, dtype=np.float64).reshape(len(df), DIGITS_N_PIXELS)
    y = df["label"].to_numpy(dtype=np.int64)
    return X, y


def load_digits_csv(path: str | Path, cache: bool = True) -> Dataset:
    """Lee digits.csv / digits_test.csv / more_digits.csv.

    Formato: columnas label (entero 0–9) e image (texto con una lista de 784
    floats en [0, 1], imagen 28×28 aplanada por filas). Devuelve X (p, 784)
    float64 e y (p,) int64.

    cache=True: guarda X, y y el sha256 del CSV en un .npz al lado del CSV y,
    en las cargas siguientes, lo usa si el sha256 coincide (si el CSV cambió, lo
    rehace).
    """
    path = Path(path)
    sha = file_sha256(path)
    cache_path = path.with_suffix(".npz")
    X = y = None
    if cache and cache_path.exists():
        with np.load(cache_path) as npz:
            if str(npz["sha256"]) == sha:
                X, y = npz["X"], npz["y"]
    from_cache = X is not None
    if not from_cache:
        X, y = _parse_digits(path)
        if cache:
            np.savez(cache_path, X=X, y=y, sha256=np.array(sha))

    return Dataset(
        X=X,
        y=y,
        feature_names=[f"px_{i}" for i in range(DIGITS_N_PIXELS)],
        target_name="label",
        meta={
            "path": str(path),
            "sha256": sha,
            "n_rows": int(len(y)),
            "image_shape": DIGITS_IMAGE_SHAPE,
            "from_cache": from_cache,
        },
    )
