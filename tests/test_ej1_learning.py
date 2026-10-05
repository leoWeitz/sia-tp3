"""F10 · Ej1 aprendizaje: configs de experiments/configs/ej1/ y analysis/ej1_learning.py.

La corrida completa usa las configs reales con un CSV sintético, pocas épocas y
dos semillas, en tmp_path: nunca toca results/ ni figures/ del repo.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.ej1_learning import (
    EXPERIMENTS,
    FIGURES,
    best_by_model,
    constant_mse,
    label_metrics,
    least_squares_mse,
    main,
    mode_label,
    mse_by_class,
    out_of_range_fraction,
    plateau,
    saturated_fraction,
)
from experiments.config import expand, load_config
from experiments.runner import run_experiment

CONFIG_DIR = Path(__file__).parents[1] / "experiments/configs/ej1"
CONFIGS = {name: CONFIG_DIR / f"{name.removeprefix('ej1_')}.json" for name in EXPERIMENTS}


# --- configs ---


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_configs_validan_y_usan_todas_las_muestras(name):
    config = load_config(CONFIGS[name])
    assert config["run_name"] == name
    ds = config["dataset"]
    assert ds["split"] == {"kind": "none"}  # R-01
    assert ds["holdout_test"] is None and ds["test_path"] is None
    assert "flagged_fraud" in ds["drop"] and "flagged_fraud" not in ds["features"]
    assert config["training"]["loss"] == "mse"
    assert len(config["seeds"]) >= 5
    assert config["logging"]["save_predictions"]


def test_configs_comparten_features_y_semillas_con_la_base():
    base = load_config(CONFIG_DIR / "base.json")
    for path in CONFIGS.values():
        config = load_config(path)
        assert config["dataset"]["features"] == base["dataset"]["features"]
        assert config["dataset"]["target"] == base["dataset"]["target"]
        assert config["seeds"] == load_config(CONFIGS["ej1_learning"])["seeds"]


def test_learning_barre_ambos_modelos_con_la_misma_grilla():
    runs = expand(load_config(CONFIGS["ej1_learning"]))
    grid = {}
    for r in runs:
        act = r["model"]["output_activation"]
        grid.setdefault(act, set()).add(
            (r["training"]["optimizer"]["lr"], r["training"]["batch_size"])
        )
    assert set(grid) == {"identity", "sigmoid"}
    assert grid["identity"] == grid["sigmoid"]
    assert {bs for _, bs in grid["identity"]} == {1, None}  # online y batch


def test_long_usa_adam_y_mas_epocas():
    learning, long = load_config(CONFIGS["ej1_learning"]), load_config(CONFIGS["ej1_long"])
    assert long["training"]["optimizer"]["kind"] == "adam"
    assert long["training"]["epochs"] >= 3 * learning["training"]["epochs"]


def test_beta_y_scaling_barren_lo_que_pide_la_spec():
    beta = load_config(CONFIGS["ej1_beta"])
    assert sorted(beta["sweep"]["model.beta"]) == [0.25, 0.5, 1.0, 2.0, 4.0]
    assert set(beta["sweep"]["dataset.normalize"]) == {"none", "zscore"}
    assert beta["model"]["output_activation"] == "sigmoid"
    scaling = load_config(CONFIGS["ej1_scaling"])
    assert set(scaling["sweep"]["dataset.normalize"]) == {"none", "minmax", "zscore"}
    assert set(scaling["sweep"]["model.output_activation"]) == {"identity", "sigmoid"}


# --- referencias y cálculos ---


def test_predictor_constante_es_la_varianza():
    assert constant_mse(np.array([0.0, 1.0])) == pytest.approx(0.25)


def test_minimos_cuadrados_exacto_e_invariante_a_escala():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3)) * [1, 100, 1e4] + [0, 5, -3]
    y = X @ np.array([0.5, -0.01, 1e-4]) + 2.0
    assert least_squares_mse(X, y) == pytest.approx(0.0, abs=1e-12)
    noisy = y + rng.normal(scale=0.1, size=len(y))
    mse = least_squares_mse(X, noisy)
    assert 0 < mse < constant_mse(noisy)
    assert least_squares_mse(X * 1000 + 7, noisy) == pytest.approx(mse)


def test_meseta():
    flat = np.concatenate([np.linspace(1.0, 0.1, 90), np.full(10, 0.1)])
    assert plateau(flat)["meseta"]
    falling = np.linspace(1.0, 0.1, 100)
    p = plateau(falling)
    assert not p["meseta"] and p["epocas_tramo"] == 10
    assert p["mejora_relativa"] == pytest.approx((falling[-11] - falling[-1]) / falling[-11])


def test_fraccion_saturada():
    # 4 s (1 − s): [1, 0.004, 0.004, 0.36] → 2 de 4 debajo de 0.05
    assert saturated_fraction(np.array([0.5, 0.999, 0.001, 0.9])) == pytest.approx(0.5)


def test_fuera_de_rango():
    assert out_of_range_fraction(np.array([-0.1, 0.5, 1.2, 1.0, 0.0])) == pytest.approx(0.4)


def test_metricas_secundarias_con_umbral():
    m = label_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.6, 0.7, 0.2]), 0.5)
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)


def test_error_por_clase():
    out = mse_by_class(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.9, 1.0]), np.array([0.1, 0.4, 0.5, 1.0])
    )
    assert out["mse_negativos"] == pytest.approx(0.02)
    assert out["mse_positivos"] == pytest.approx(0.08)
    assert out["fraccion_error_positivos"] == pytest.approx(0.8)
    assert "error_por_clase" in json.loads(json.dumps({"error_por_clase": out}))


def test_modo():
    assert mode_label(1) == "online"
    assert mode_label(None) == "batch" and mode_label(float("nan")) == "batch"
    assert mode_label(32) == "mini-batch 32"


def test_mejor_config_ignora_las_que_divergieron():
    table = pd.DataFrame(
        {
            "model.output_activation": ["identity", "identity", "sigmoid"],
            "hash": ["a", "b", "c"],
            "n": [5, 5, 5],
            "n_ok": [4, 5, 5],
            "mse_mean": [0.01, 0.02, 0.03],
        }
    )
    best = best_by_model(table)
    assert best["identity"]["hash"] == "b" and best["sigmoid"]["hash"] == "c"
    with pytest.raises(ValueError, match="convergió"):
        best_by_model(table.assign(n_ok=0))


# --- corrida completa ---


def _synthetic_csv(path: Path, n: int = 300) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "timestamp": rng.integers(1_700_000_000, 1_731_000_000, n),
            "amount_usd": np.round(rng.lognormal(4, 1, n), 2),
            "quantity_purchased": rng.integers(1, 25, n),
            "session_duration_seconds": np.round(rng.uniform(5, 700, n), 1),
            "days_since_last_purchase": np.round(rng.exponential(14, n), 2),
            "account_age_days": rng.integers(1, 3650, n),
            "device_screen_resolution": rng.integers(1_000_000, 8_300_000, n),
            "time_since_last_login_s": np.round(rng.exponential(3600, n), 1),
            "items_viewed_before_purchase": rng.integers(1, 30, n),
        }
    )
    h = (df["quantity_purchased"] - 12) / 4 - (df["account_age_days"] - 1800) / 1500
    df["big_model_fraud_probability"] = 1 / (1 + np.exp(-h))
    df["flagged_fraud"] = (df["big_model_fraud_probability"] > 0.85).astype(int)
    df.to_csv(path, index=False)


def _small_config(name: str, csv: Path, out: Path) -> Path:
    config = json.loads(CONFIGS[name].read_text(encoding="utf-8"))
    config["dataset"]["path"] = str(csv)
    config["seeds"] = [0, 1]
    config["training"]["epochs"] = 10 if name == "ej1_long" else 5
    sweep = config.get("sweep", {})
    if name == "ej1_learning":
        sweep["training.optimizer.lr"] = [0.001, 0.01]
    if name == "ej1_beta":
        sweep["model.beta"] = [0.5, 1.0]
    path = out / f"{name}.json"
    path.write_text(json.dumps(config))
    return path


def test_main_genera_todas_las_figuras(tmp_path):
    csv = tmp_path / "fraud.csv"
    _synthetic_csv(csv)
    results = tmp_path / "results"
    for name in EXPERIMENTS:
        run_experiment(
            _small_config(name, csv, tmp_path), results_dir=results, write=lambda _: None
        )
    out = tmp_path / "figs"
    assert main(["--results-dir", str(results), "--out-dir", str(out)]) == 0
    for name in FIGURES:
        assert (out / f"{name}.png").exists(), name
        assert (out / f"{name}.pdf").exists(), name
    resumen = json.loads((out / "resumen.json").read_text(encoding="utf-8"))
    assert set(resumen["mejor"]) == {"lineal", "logística"}
    assert resumen["referencias"]["minimos_cuadrados"] <= resumen["referencias"]["constante"]
    assert len(pd.read_csv(out / "tabla.csv")) == 4


def test_main_sin_corridas_devuelve_2(tmp_path):
    assert main(["--results-dir", str(tmp_path), "--out-dir", str(tmp_path / "o")]) == 2
