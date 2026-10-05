"""Arma datasets/digits_union.csv: digits.csv + more_digits.csv sin imágenes repetidas.

La cátedra entrega digits.csv y more_digits.csv por separado. En el Ej3 (F13) entrenamos
con la unión de los dos, sacando las imágenes que aparecen en ambos (3 689): si no se
sacan, la misma imagen puede caer en train y en validación a la vez.

Reglas:
- Se concatenan primero las filas de digits.csv y después las de more_digits.csv.
- Dos filas son la misma imagen si tienen los mismos 784 píxeles; se queda la primera.
- Las filas se copian tal cual (sin reformatear números) y se escriben con fin de línea
  LF. Así el archivo es idéntico, byte a byte, al que usaron las corridas de
  results/ej3_* (su sha256 está en cada metrics.json, clave data.sha256).
- Si está datasets/digits_test.csv, verifica que ninguna imagen del test esté en la unión.

Uso:
    python -m data.build_digits_union [--datasets-dir datasets]
"""

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

HEADER = b"label,image"
EXPECTED_ROWS = 24501
EXPECTED_SHA256 = "0d7c968789649942fad24fbce0160f7937a0be243e583514d8f11c1e1bc4af63"


def read_rows(path: str | Path) -> list[bytes]:
    """Filas de datos de un CSV de dígitos (sin el encabezado), como bytes crudos.

    Acepta fin de línea CRLF o LF. Falla si el encabezado no es "label,image".
    """
    lines = Path(path).read_bytes().splitlines()
    if not lines or lines[0].strip() != HEADER:
        raise ValueError(f"{path}: el encabezado tiene que ser {HEADER.decode()!r}")
    return [line for line in lines[1:] if line.strip()]


def split_row(row: bytes) -> tuple[int, bytes]:
    """(etiqueta, texto de la imagen sin comillas) de una fila 'label,"[...]"'."""
    label, image = row.split(b",", 1)
    return int(label), image.strip().strip(b'"')


def pixel_key(row: bytes) -> bytes:
    """Clave de la imagen: sus 784 píxeles como float64 (independiente del formato)."""
    _, image = split_row(row)
    pixels = np.asarray(json.loads(image), dtype=np.float64)
    if pixels.shape != (784,):
        raise ValueError(f"cada imagen tiene que tener 784 píxeles, llegó {pixels.shape}")
    return pixels.tobytes()


def build_union(first: Sequence[bytes], second: Sequence[bytes]) -> tuple[list[bytes], dict]:
    """Concatena first y second sin imágenes repetidas (se queda la primera aparición).

    Devuelve (filas, info) con info = {"duplicates": filas descartadas,
    "label_conflicts": repetidas con otra etiqueta, "keys": claves de las filas}.
    """
    seen: dict[bytes, int] = {}
    rows: list[bytes] = []
    keys: list[bytes] = []
    duplicates = conflicts = 0
    for row in [*first, *second]:
        key = pixel_key(row)
        label = split_row(row)[0]
        if key in seen:
            duplicates += 1
            conflicts += int(seen[key] != label)
            continue
        seen[key] = label
        rows.append(row)
        keys.append(key)
    return rows, {"duplicates": duplicates, "label_conflicts": conflicts, "keys": keys}


def to_csv_bytes(rows: Sequence[bytes]) -> bytes:
    """Encabezado + filas, con fin de línea LF y salto final."""
    return HEADER + b"\n" + b"\n".join(rows) + b"\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets-dir", default="datasets", type=Path)
    parser.add_argument(
        "--out", default=None, type=Path, help="default: <datasets-dir>/digits_union.csv"
    )
    args = parser.parse_args(argv)
    base: Path = args.datasets_dir
    out: Path = args.out or base / "digits_union.csv"

    digits = read_rows(base / "digits.csv")
    more = read_rows(base / "more_digits.csv")
    rows, info = build_union(digits, more)

    test_path = base / "digits_test.csv"
    if test_path.exists():
        test_keys = {pixel_key(row) for row in read_rows(test_path)}
        leaked = sum(key in test_keys for key in info["keys"])
        if leaked:
            print(f"ERROR: {leaked} imágenes de digits_test.csv están en la unión")
            return 1
        test_msg = "ninguna imagen de digits_test.csv está en la unión"
    else:
        test_msg = "no está digits_test.csv: no se verificó la fuga hacia test"

    data = to_csv_bytes(rows)
    out.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    counts = Counter(split_row(row)[0] for row in rows)

    print(f"digits.csv: {len(digits)} filas · more_digits.csv: {len(more)} filas")
    print(
        f"duplicados descartados: {info['duplicates']} "
        f"(con otra etiqueta: {info['label_conflicts']})"
    )
    print(f"{out}: {len(rows)} filas · sha256 {sha}")
    print("por dígito: " + ", ".join(f"{d}: {counts[d]}" for d in sorted(counts)))
    print(test_msg)
    if len(rows) != EXPECTED_ROWS or sha != EXPECTED_SHA256:
        print(
            "ATENCIÓN: no coincide con el archivo de las corridas de results/ej3_* "
            f"({EXPECTED_ROWS} filas, sha256 {EXPECTED_SHA256})"
        )
        return 1
    print("OK: idéntico al archivo usado en results/ej3_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
