"""F12 · Tests de la EDA de dígitos (analysis/ej2_eda.py).

Verifica cálculos de balance de clases, dígitos promedio, estadísticos de píxeles,
detección de duplicados y corrida completa del script sobre datos sintéticos en tmp_path.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.ej2_eda import (
    FIGURES,
    check_duplicates,
    class_balance_table,
    compute_average_digits,
    main,
    pixel_stats,
)
from data.loaders import DIGITS_IMAGE_SHAPE, DIGITS_N_PIXELS

ROOT = Path(__file__).parents[1]
DIGITS_CSV = ROOT / "datasets/digits.csv"
needs_digits = pytest.mark.skipif(not DIGITS_CSV.exists(), reason="falta datasets/digits.csv")


def synthetic_digits_csv(path: Path, n_samples: int = 50, seed: int = 0) -> None:
    """Genera un CSV con columnas label e image de tamaño DIGITS_N_PIXELS."""
    rng = np.random.default_rng(seed)
    # Excluimos clase 8 a propósito para reproducir el escenario del dataset
    labels = rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 9], size=n_samples)
    images = []
    for _ in range(n_samples):
        # 784 píxeles en [0, 1]
        px = np.round(rng.uniform(0.0, 1.0, DIGITS_N_PIXELS), 4)
        images.append(json.dumps(px.tolist()))

    df = pd.DataFrame({"label": labels, "image": images})
    df.to_csv(path, index=False)


# --- Tests de funciones auxiliares ---


def test_class_balance_table_calcula_conteos_y_porcentajes():
    y_tr = np.array([0, 1, 1, 2, 2, 2])
    y_te = np.array([0, 1, 8])

    df = class_balance_table(y_tr, y_te, n_classes=10)
    assert len(df) == 10
    assert df.loc[0, "train_n"] == 1
    assert df.loc[1, "train_n"] == 2
    assert df.loc[2, "train_n"] == 3
    assert df.loc[8, "train_n"] == 0
    assert df.loc[8, "train_pct"] == 0.0

    assert df.loc[8, "test_n"] == 1
    assert np.isclose(df["train_pct"].sum(), 100.0)


def test_compute_average_digits_devuelve_none_para_clases_vacias():
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (10, DIGITS_N_PIXELS))
    # Clases 0, 1, ..., 7, 9 (falta 8)
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 0])

    averages = compute_average_digits(X, y, n_classes=10)
    assert len(averages) == 10
    assert averages[8] is None
    assert averages[0] is not None
    assert averages[0].shape == DIGITS_IMAGE_SHAPE


def test_pixel_stats_identifica_pixeles_sin_varianza():
    X = np.zeros((20, DIGITS_N_PIXELS))
    # Hacemos que solo 100 píxeles tengan señal
    X[:, :100] = np.random.default_rng(0).uniform(0.1, 1.0, (20, 100))

    stats = pixel_stats(X)
    assert stats["min"] == 0.0
    assert stats["max"] > 0.0
    assert stats["zero_var_pixels"] == DIGITS_N_PIXELS - 100
    assert stats["active_pixels"] == 100


def test_check_duplicates_detecta_duplicados_intra_e_inter():
    row1 = np.ones(DIGITS_N_PIXELS)
    row2 = np.zeros(DIGITS_N_PIXELS)
    row3 = np.ones(DIGITS_N_PIXELS) * 0.5

    d1 = np.array([row1, row2, row1])  # 1 duplicado intra
    d2 = np.array([row1, row3])        # 1 coincidencia con d1

    dups = check_duplicates({"d1": d1, "d2": d2})
    assert dups["intra_dataset"]["d1"] == 1
    assert dups["intra_dataset"]["d2"] == 0
    assert dups["inter_dataset"]["d1_vs_d2"] == 1


# --- Corrida completa del script sobre datos sintéticos ---


def test_main_ejecuta_correctamente_en_tmp_path(tmp_path):
    tr_path = tmp_path / "digits_tr.csv"
    te_path = tmp_path / "digits_te.csv"
    out_dir = tmp_path / "eda_out"

    synthetic_digits_csv(tr_path, n_samples=30, seed=1)
    synthetic_digits_csv(te_path, n_samples=15, seed=2)

    code = main([
        "--train", str(tr_path),
        "--test", str(te_path),
        "--out-dir", str(out_dir),
    ])
    assert code == 0

    # Verificar que existan todas las figuras pedidas
    for fig in FIGURES:
        assert (out_dir / f"{fig}.png").exists(), f"Falta {fig}.png"
        assert (out_dir / f"{fig}.pdf").exists(), f"Falta {fig}.pdf"

    # Verificar tablas y resumen
    assert (out_dir / "class_balance.csv").exists()
    assert (out_dir / "pixel_stats.csv").exists()
    assert (out_dir / "resumen.json").exists()

    resumen = json.loads((out_dir / "resumen.json").read_text(encoding="utf-8"))
    assert resumen["datasets"]["train"]["n_rows"] == 30
    assert resumen["datasets"]["test"]["n_rows"] == 15
    assert "conclusions" in resumen


def test_main_sin_train_retorna_codigo_error(tmp_path):
    code = main(["--train", str(tmp_path / "inexistente.csv"), "--out-dir", str(tmp_path)])
    assert code == 2


@needs_digits
def test_dataset_real_reproduce_hallazgos_clave():
    from data.loaders import load_digits_csv

    ds = load_digits_csv(DIGITS_CSV)
    counts = np.bincount(ds.y, minlength=10)

    # Regla dura: 0 muestras de 8 en digits.csv
    assert counts[8] == 0
    # Desbalance severo del 5
    assert counts[5] == 271
    # Total de filas
    assert len(ds.y) == 12449
    # Píxeles en [0, 1]
    assert ds.X.min() >= 0.0 and ds.X.max() <= 1.0
