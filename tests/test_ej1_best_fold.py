"""F11 · experiments/ej1_best_fold.py: pseudo-test en DESARROLLO, mejor fold vs todo el resto.

Todo corre sobre un CSV sintético en tmp_path, con el test y los resultados
redirigidos ahí: nunca toca results/ del repo.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.config import load_config
from experiments.ej1_best_fold import (
    PSEUDO_TEST_RATIO,
    main,
    pseudo_test_indices,
    run_best_fold,
)

CONFIG = Path(__file__).parents[1] / "experiments/configs/ej1/best_fold.json"


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


def _small_config(tmp_path: Path, seeds=(0, 1)) -> Path:
    csv = tmp_path / "fraud.csv"
    if not csv.exists():
        _synthetic_csv(csv)
    config = json.loads(CONFIG.read_text())
    config["dataset"]["path"] = str(csv)
    config["dataset"]["holdout_test"]["indices_path"] = str(tmp_path / "split/test_idx.npy")
    config["seeds"] = list(seeds)
    config["training"]["epochs"] = 5
    path = tmp_path / "best_fold.json"
    path.write_text(json.dumps(config))
    return path


def _quiet(_: str) -> None:
    pass


def test_config_real_valida():
    config = load_config(CONFIG)
    assert config["run_name"] == "ej1_best_fold"
    assert config["dataset"]["split"]["kind"] == "kfold"
    assert config["dataset"]["holdout_test"]["indices_path"] == "results/ej1_split/test_idx.npy"
    assert len(config["seeds"]) >= 5  # spec: ≥ 5 semillas de partición
    assert config["logging"]["save_model"]


def test_pseudo_test_dentro_de_desarrollo_y_estratificado():
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 1, 1000)
    test_idx = np.sort(rng.choice(1000, 200, replace=False))
    pseudo = pseudo_test_indices(y, test_idx, seed=3, task="regression")
    dev = np.setdiff1d(np.arange(1000), test_idx)
    assert np.intersect1d(pseudo, test_idx).size == 0
    assert np.isin(pseudo, dev).all()
    assert len(pseudo) == pytest.approx(PSEUDO_TEST_RATIO * len(dev), abs=10)
    assert np.all(np.diff(pseudo) > 0)
    # Mismo seed → misma partición; otro seed → otra.
    assert np.array_equal(pseudo, pseudo_test_indices(y, test_idx, seed=3, task="regression"))
    assert not np.array_equal(pseudo, pseudo_test_indices(y, test_idx, seed=4, task="regression"))
    # Estratificada por deciles: la media del target se conserva.
    assert y[pseudo].mean() == pytest.approx(y[dev].mean(), abs=0.03)


def test_particion_excluye_test_y_pseudo_test_de_todo_entrenamiento(tmp_path):
    config = _small_config(tmp_path, seeds=(0,))
    root = tmp_path / "results"
    [summary] = run_best_fold(config, results_dir=root, write=_quiet)
    out = root / "ej1_best_fold"
    part = out / "partition_s0"
    test_idx = np.load(tmp_path / "split/test_idx.npy")
    pseudo = np.load(part / "pseudo_idx.npy")
    assert np.intersect1d(test_idx, pseudo).size == 0
    assert summary["n_pseudo_test"] == len(pseudo)
    assert len(summary["folds"]) == 5

    val_union = []
    for fold in summary["folds"]:
        z = np.load(out / fold["run"] / "predictions.npz")
        val_union.append(z["idx"])
        assert np.intersect1d(z["idx"], pseudo).size == 0
        assert np.intersect1d(z["idx"], test_idx).size == 0
        m = json.loads((out / fold["run"] / "metrics.json").read_text())
        assert m["data"]["n_test_excluded"] == len(test_idx) + len(pseudo)
    # Los folds cubren exactamente el resto de DESARROLLO.
    rest = np.setdiff1d(np.arange(300), np.union1d(test_idx, pseudo))
    assert np.array_equal(np.sort(np.concatenate(val_union)), rest)

    m_rest = json.loads((out / summary["rest"]["run"] / "metrics.json").read_text())
    assert m_rest["data"]["n_train"] == len(rest)
    assert m_rest["val"] is None
    assert summary["rest"]["epochs"] >= 1

    for name in [f"pseudo_f{f['fold']}.npz" for f in summary["folds"]] + ["pseudo_rest.npz"]:
        z = np.load(part / name)
        assert np.array_equal(z["idx"], pseudo)
        assert z["y_score"].shape == (len(pseudo), 1)


def test_mejor_fold_es_el_de_menor_val_loss(tmp_path):
    [summary] = run_best_fold(
        _small_config(tmp_path, seeds=(1,)), results_dir=tmp_path / "r", write=_quiet
    )
    losses = [f["val_loss"] for f in summary["folds"]]
    assert summary["best_fold"] == summary["folds"][int(np.argmin(losses))]["fold"]
    saved = json.loads((tmp_path / "r/ej1_best_fold/partition_s1/partition.json").read_text())
    assert saved["best_fold"] == summary["best_fold"]
    for f in [*saved["folds"], saved["rest"]]:
        assert {"mse", "rmse", "mae"} <= set(f["pseudo_test"])


def _strip(summary: dict) -> dict:
    """El resumen sin lo que depende de la carpeta o del reloj.

    Los run ids también dependen de la carpeta: el hash incluye la ruta de
    excluded_idx.npy (holdout_test.indices_path).
    """
    s = json.loads(json.dumps(summary))
    for key in ("pseudo_idx_path", "excluded_idx_path", "seconds"):
        s.pop(key, None)
    for f in [*s["folds"], s["rest"]]:
        f.pop("run")
    return s


def test_reproducible_con_la_misma_semilla(tmp_path):
    config = _small_config(tmp_path, seeds=(2,))
    [a] = run_best_fold(config, results_dir=tmp_path / "a", write=_quiet)
    [b] = run_best_fold(config, results_dir=tmp_path / "b", write=_quiet)
    assert _strip(a) == _strip(b)


def test_relanzar_saltea_particiones_completas(tmp_path):
    config = _small_config(tmp_path, seeds=(0,))
    run_best_fold(config, results_dir=tmp_path / "r", write=_quiet)
    lines = []
    run_best_fold(config, results_dir=tmp_path / "r", write=lines.append)
    assert any("salteada" in line for line in lines)


def test_main_valida_la_config(tmp_path):
    bad = json.loads(_small_config(tmp_path).read_text())
    bad["dataset"]["split"] = {"kind": "holdout"}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    assert main([str(path), "--results-dir", str(tmp_path / "r")]) == 2
    bad["dataset"]["split"] = {"kind": "kfold"}
    bad["dataset"]["holdout_test"] = None
    path.write_text(json.dumps(bad))
    assert main([str(path), "--results-dir", str(tmp_path / "r")]) == 2


def test_smoke_va_a_results_smoke(tmp_path):
    config = _small_config(tmp_path, seeds=(0, 1))
    assert main([str(config), "--smoke", "--results-dir", str(tmp_path / "r")]) == 0
    assert (tmp_path / "r/_smoke/ej1_best_fold/partition_s0/partition.json").exists()
    assert not (tmp_path / "r/_smoke/ej1_best_fold/partition_s1").exists()
    assert not (tmp_path / "r/ej1_best_fold").exists()
