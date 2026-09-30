"""F09 · EDA del Ej1: config base y analysis/ej1_eda.py.

Los tests con el CSV real se saltean si no está en datasets/ (el CI no lo
tiene). La corrida completa del script usa un CSV sintético en tmp_path:
nunca toca figures/ del repo.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.ej1_eda import (
    COLUMNS,
    FIGURES,
    LABEL,
    TARGET,
    class_bounds,
    decile_profile,
    main,
    ranges_table,
    resolution_clusters,
    separation,
    spikes,
    strata_balance,
)
from experiments.config import load_config
from experiments.runner import load_dataset

ROOT = Path(__file__).parents[1]
BASE_CONFIG = ROOT / "experiments/configs/ej1/base.json"
FRAUD_CSV = ROOT / "datasets/fraud_dataset.csv"
needs_csv = pytest.mark.skipif(not FRAUD_CSV.exists(), reason="falta datasets/fraud_dataset.csv")


def synthetic_fraud(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Dataset con las columnas de fraud_dataset.csv y una regla de fraude conocida."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "timestamp": rng.integers(1_700_000_000, 1_731_000_000, n),
            "amount_usd": np.round(rng.lognormal(4, 1, n), 2),
            "quantity_purchased": rng.integers(1, 25, n),
            "session_duration_seconds": np.round(rng.uniform(5, 700, n), 1),
            "days_since_last_purchase": np.round(rng.exponential(14, n), 2),
            "account_age_days": rng.integers(1, 3650, n),
            "device_screen_resolution": rng.choice([1049088, 2073600], n)
            + rng.integers(-5000, 5000, n),
            "time_since_last_login_s": np.round(rng.exponential(3600, n), 1),
            "items_viewed_before_purchase": rng.integers(1, 30, n),
        }
    )
    df.loc[: n // 10, "amount_usd"] = 100.0
    prob = 1 / (1 + np.exp(-(df["quantity_purchased"] - 12) / 2))
    df["big_model_fraud_probability"] = prob
    df["flagged_fraud"] = (prob > 0.85).astype(int)
    return df[list(COLUMNS)]


# --- experiments/configs/ej1/base.json ---


def test_config_base_pasa_load_config():
    config = load_config(BASE_CONFIG)
    assert config["run_name"] == "ej1_base"
    assert len(config["seeds"]) >= 3


def test_flagged_fraud_nunca_se_usa_para_entrenar():
    ds = load_config(BASE_CONFIG)["dataset"]
    assert LABEL in ds["drop"]
    assert LABEL not in ds["features"]
    assert ds["target"] != LABEL


def test_target_es_la_probabilidad_de_bigmodel():
    config = load_config(BASE_CONFIG)
    assert config["dataset"]["target"] == TARGET
    assert config["dataset"]["task"] == "regression"


def test_config_base_usa_todas_las_muestras_y_salida_en_0_1():
    config = load_config(BASE_CONFIG)
    assert config["dataset"]["split"] == {"kind": "none"}  # R-01
    assert config["dataset"]["holdout_test"] is None
    assert config["model"]["output_activation"] == "sigmoid"  # R-05
    assert config["model"]["layers"] == ["auto", 1]


def test_features_y_drop_son_columnas_del_dataset_y_no_se_pisan():
    ds = load_config(BASE_CONFIG)["dataset"]
    assert set(ds["features"]) <= set(COLUMNS)
    assert set(ds["drop"]) <= set(COLUMNS)
    assert not set(ds["features"]) & set(ds["drop"])
    assert set(ds["features"]) | set(ds["drop"]) | {TARGET} == set(COLUMNS)


@needs_csv
def test_columnas_del_config_cargan_con_load_csv():
    ds = load_config(BASE_CONFIG)["dataset"]
    data = load_dataset(ds, path=str(FRAUD_CSV))
    assert data.X.shape == (7500, len(ds["features"]))
    assert data.feature_names == ds["features"]
    assert np.isfinite(data.X).all() and np.isfinite(data.y).all()
    assert data.y.min() >= 0.0 and data.y.max() <= 1.0


@needs_csv
def test_csv_real_tiene_las_columnas_documentadas():
    assert tuple(pd.read_csv(FRAUD_CSV, nrows=1).columns) == COLUMNS


# --- helpers ---


def test_tabla_de_rangos_coincide_con_numpy():
    df = synthetic_fraud()
    table = ranges_table(df)
    col = df["amount_usd"].to_numpy()
    row = table.loc["amount_usd"]
    assert row["min"] == col.min() and row["max"] == col.max()
    np.testing.assert_allclose(
        row[["p1", "p50", "p99"]].astype(float), np.percentile(col, [1, 50, 99])
    )
    assert table.loc["flagged_fraud", "tipo"] == "binaria"
    assert table.loc["quantity_purchased", "tipo"] == "entera"
    assert table.loc["amount_usd", "tipo"] == "continua"
    assert list(table.index) == list(COLUMNS)


def test_separacion_perfecta_y_con_solapamiento():
    df = pd.DataFrame({TARGET: [0.1, 0.4, 0.9, 0.95], LABEL: [0, 0, 1, 1]})
    sep = separation(df)
    assert sep["max_negativo"] == 0.4 and sep["min_positivo"] == 0.9
    assert sep["separable"] and sep["n_solapados"] == 0
    df.loc[1, TARGET] = 0.92
    sep = separation(df)
    assert not sep["separable"] and sep["n_solapados"] == 2


def test_cotas_por_clase_detectan_reglas_duras():
    df = synthetic_fraud()
    bounds = class_bounds(df)["quantity_purchased"]
    # en el sintético, fraude ⇔ quantity > 12 + 2·ln(0.85/0.15) ≈ 15.5
    assert bounds["neg_max"] <= 15 and bounds["pos_min"] >= 16
    assert bounds["pos_fuera_de_rango_neg"] == int(df[LABEL].sum())


def test_spikes_encuentra_valores_exactos_repetidos():
    df = synthetic_fraud()
    found = spikes(df, min_count=20)
    hit = [s for s in found if s["columna"] == "amount_usd" and s["valor"] == 100.0]
    assert len(hit) == 1
    assert hit[0]["n"] >= 41
    # las columnas enteras de pocos valores no son "spikes"
    assert all(s["columna"] != "quantity_purchased" for s in found)


def test_clusters_de_resolucion_asigna_la_mas_cercana():
    values = np.array([1049088 + 3000, 2073600 - 4000, 8294400 + 10, 1024000 - 2000])
    out = resolution_clusters(values)
    assert list(out["resolucion"]) == ["1366x768", "1920x1080", "3840x2160", "1280x800"]
    np.testing.assert_array_equal(out["desvio_px"], [3000, -4000, 10, -2000])


def test_perfil_por_deciles():
    df = synthetic_fraud()
    prof = decile_profile(df, "quantity_purchased")
    assert len(prof) == 10
    assert prof[TARGET].is_monotonic_increasing  # el target sube con quantity
    assert prof["n"].sum() == len(df)


def test_balance_de_estratos_por_cuantiles():
    df = synthetic_fraud()
    out = strata_balance(df)
    assert out["n"].sum() == len(df)
    assert out["tasa_positivos"].between(0, 1).all()


# --- corrida completa ---


def test_main_genera_todas_las_figuras_y_resumen(tmp_path):
    csv = tmp_path / "fraud.csv"
    synthetic_fraud().to_csv(csv, index=False)
    out = tmp_path / "eda"
    assert main(["--data", str(csv), "--out-dir", str(out)]) == 0
    for name in FIGURES:
        assert (out / f"{name}.png").exists(), name
        assert (out / f"{name}.pdf").exists(), name
    assert (out / "rangos.csv").exists()
    resumen = json.loads((out / "resumen.json").read_text())
    for key in (
        "sha256",
        "filas",
        "columnas",
        "tasa_positivos",
        "bigmodel",
        "separacion",
        "cotas_por_clase",
    ):
        assert key in resumen
    assert resumen["filas"] == 400


def test_main_sin_csv_devuelve_2(tmp_path):
    assert main(["--data", str(tmp_path / "no.csv"), "--out-dir", str(tmp_path)]) == 2
