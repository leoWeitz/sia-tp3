"""F11 · Ej1 generalización y umbral: configs de ej1/ y analysis/ej1_generalization.py.

La corrida completa usa las configs reales con un CSV sintético, pocas épocas y
pocas semillas, con el test y los resultados en tmp_path: nunca toca results/
ni figures/ del repo.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.ej1_generalization import (
    FIGURES,
    FINAL_FIGURES,
    choose_threshold,
    cost_sensitivity,
    criteria_table,
    fig_split_strategy,
    main,
    near_cut_fraction,
    oof_predictions,
    per_1000,
    select_config,
    split_estimates,
    strategy_name,
)
from experiments.config import config_hash, expand, load_config
from experiments.ej1_best_fold import run_best_fold
from experiments.runner import final_eval, run_experiment

CONFIG_DIR = Path(__file__).parents[1] / "experiments/configs/ej1"
F11_CONFIGS = ("cv", "split_strategy", "best_fold")
TEST_IDX = "results/ej1_split/test_idx.npy"


# --- configs ---


@pytest.mark.parametrize("name", F11_CONFIGS)
def test_configs_usan_el_mismo_test_y_nunca_la_etiqueta(name):
    config = load_config(CONFIG_DIR / f"{name}.json")
    ds = config["dataset"]
    assert ds["holdout_test"] == {
        "ratio": 0.2,
        "seed": 0,
        "stratified": True,
        "indices_path": TEST_IDX,
    }
    assert ds["test_path"] is None
    assert "flagged_fraud" in ds["drop"] and "flagged_fraud" not in ds["features"]
    assert ds["normalize"] == "zscore" and ds["task"] == "regression"
    assert config["training"]["loss"] == "mse"
    assert config["training"]["early_stopping"]["monitor"] == "val_loss"
    assert config["logging"]["save_predictions"]
    assert len(config["seeds"]) >= 3


def test_configs_comparten_features_con_la_base():
    base = load_config(CONFIG_DIR / "base.json")["dataset"]
    for name in F11_CONFIGS:
        ds = load_config(CONFIG_DIR / f"{name}.json")["dataset"]
        assert ds["features"] == base["features"] and ds["target"] == base["target"]


def test_cv_es_kfold_estratificado_con_particion_por_semilla():
    config = load_config(CONFIG_DIR / "cv.json")
    assert config["dataset"]["split"] == {"kind": "kfold", "k": 5, "stratified": True, "seed": None}
    runs = expand(config)
    assert {r["fold"] for r in runs} == set(range(5))


def test_split_strategy_barre_las_tres_estrategias_y_las_fracciones():
    config = load_config(CONFIG_DIR / "split_strategy.json")
    names = {strategy_name(s) for s in config["sweep"]["dataset.split"]}
    assert {"holdout aleatorio", "holdout estratificado", "5-fold estratificado"} <= names
    ratios = {
        s["ratio"]
        for s in config["sweep"]["dataset.split"]
        if s["kind"] == "holdout" and s["stratified"]
    }
    assert ratios == {0.5, 0.6, 0.7, 0.8, 0.9}
    assert len(config["seeds"]) >= 20


def test_split_strategy_y_best_fold_usan_el_mismo_modelo():
    a = load_config(CONFIG_DIR / "split_strategy.json")
    b = load_config(CONFIG_DIR / "best_fold.json")
    assert a["model"] == b["model"] and a["training"] == b["training"]


# --- cálculos ---


def _cv_runs():
    return pd.DataFrame(
        {
            "hash": ["a", "a", "b", "b", "c", "c"],
            "status": ["ok", "ok", "ok", "ok", "ok", "diverged"],
            "val.loss": [0.02, 0.03, 0.04, 0.05, 0.001, 0.001],
        }
    )


def test_select_config_minimo_val_loss_entre_las_que_no_divergieron():
    assert select_config(_cv_runs()) == "a"


def test_oof_cubre_desarrollo_una_vez_por_semilla(tmp_path):
    runs = []
    idx = np.arange(20)
    for seed in (0, 1):
        perm = np.random.default_rng(seed).permutation(idx)
        for fold, part in enumerate(np.array_split(perm, 4)):
            name = f"h_s{seed}_f{fold}"
            (tmp_path / name).mkdir()
            np.savez(
                tmp_path / name / "predictions.npz",
                idx=part,
                y_true=part[:, None] / 20.0,
                y_score=part[:, None] / 21.0,
            )
            runs.append({"run": name, "hash": "h", "seed": seed, "fold": fold})
    df = oof_predictions(tmp_path, pd.DataFrame(runs), "h")
    assert len(df) == 40
    for _, g in df.groupby("seed"):
        assert sorted(g["idx"]) == list(idx)
    assert np.allclose(df["y_true"], df["idx"] / 20.0)

    # Un fold repetido rompe la cobertura: error, no un OOF silenciosamente sesgado.
    runs[1] = {**runs[1], "run": runs[0]["run"]}
    with pytest.raises(ValueError):
        oof_predictions(tmp_path, pd.DataFrame(runs), "h")


def test_per_1000():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.2, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    r = per_1000(y, s, 0.5)
    assert r == pytest.approx(
        {
            "fraudes": 200.0,
            "detectados": 100.0,
            "no_detectados": 100.0,
            "falsas_alarmas": 100.0,
            "alertas": 200.0,
        }
    )


def _toy_scores(n: int = 2000):
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=n) < 0.12).astype(int)
    s = np.clip(0.25 + 0.5 * y + rng.normal(0, 0.2, n), 0, 1)
    return y, s


def test_errores_cerca_del_corte():
    prob = np.array([0.1, 0.8, 0.9, 0.99, 0.5])
    y = prob > 0.85
    flagged = np.array([False, True, False, True, True])  # errores en 0.8, 0.9 y 0.5
    assert near_cut_fraction(prob, y, flagged) == pytest.approx(2 / 3)
    assert np.isnan(near_cut_fraction(prob, y, y))


def test_criterios():
    y, s = _toy_scores()
    table = criteria_table(y, s, cost_ratio=5)
    assert list(table["criterio"]) == [
        "f1",
        "f2",
        "youden",
        "cost:5",
        "precision_at_recall:0.8",
        "precision_at_recall:0.9",
    ]
    row = table.set_index("criterio")
    assert row.loc["precision_at_recall:0.9", "recall"] >= 0.9
    assert row.loc["precision_at_recall:0.8", "recall"] >= 0.8
    # F2 pesa más el recall que F1: su umbral nunca es mayor.
    assert row.loc["f2", "threshold"] <= row.loc["f1", "threshold"]
    assert choose_threshold(y, s, "f1")["threshold"] == row.loc["f1", "threshold"]
    with pytest.raises(ValueError):
        choose_threshold(y, s, "accuracy")


def test_costo_mas_caro_el_fn_baja_el_umbral():
    y, s = _toy_scores()
    table = cost_sensitivity(y, s, ratios=(1, 2, 5, 10, 20))
    assert list(table["ratio"]) == [1, 2, 5, 10, 20]
    assert np.all(np.diff(table["threshold"]) <= 0)
    assert np.all(np.diff(table["recall"]) >= 0)


def test_strategy_name():
    assert strategy_name({"kind": "holdout", "ratio": 0.8, "stratified": False}) == (
        "holdout aleatorio"
    )
    assert strategy_name({"kind": "holdout", "ratio": 0.6, "stratified": True}) == (
        "holdout estratificado"
    )
    assert strategy_name({"kind": "kfold", "k": 5, "stratified": True}) == "5-fold estratificado"


def test_split_estimates_promedia_los_folds_de_cada_semilla(tmp_path):
    labels = np.array([0, 1] * 10)
    rows = []
    for seed in (0, 1):
        for fold, part in enumerate(np.array_split(np.arange(20), 2)):
            name = f"k_s{seed}_f{fold}"
            (tmp_path / name).mkdir()
            score = labels[part] * 0.5 + 0.2 * fold
            np.savez(
                tmp_path / name / "predictions.npz",
                idx=part,
                y_true=labels[part][:, None].astype(float),
                y_score=score[:, None],
            )
            rows.append(
                {
                    "run": name,
                    "hash": "k",
                    "seed": seed,
                    "fold": fold,
                    "dataset.split.kind": "kfold",
                    "dataset.split.k": 2,
                    "dataset.split.stratified": True,
                    "dataset.split.ratio": np.nan,
                    "val.mse": 0.1 * (fold + 1),
                    "train.mse": 0.05,
                }
            )
    partitions, estimates = split_estimates(tmp_path, pd.DataFrame(rows), labels)
    assert len(partitions) == 4 and len(estimates) == 2
    assert estimates["mse"].tolist() == pytest.approx([0.15, 0.15])
    assert set(partitions["fraud_rate"]) == {0.5}
    assert set(estimates["strategy"]) == {"2-fold estratificado"}


def test_fig_split_strategy_muestra_el_mse_de_validacion_por_semilla():
    import matplotlib.pyplot as plt

    strategies = {
        "holdout aleatorio": 0.8,
        "holdout estratificado": 0.8,
        "5-fold estratificado": None,
    }
    rows = [
        {
            "strategy": strategy,
            "ratio": np.nan if ratio is None else ratio,
            "seed": seed,
            "fraud_rate": 0.10 + 0.01 * seed,
            "ap": 0.95,
            "mse": 0.011 + 0.001 * seed + 0.0001 * i,
            "train_mse": 0.011,
        }
        for i, (strategy, ratio) in enumerate(strategies.items())
        for seed in range(3)
    ]
    # Una fracción de entrenamiento distinta de 80/20 no entra a esta figura.
    rows.append({**rows[3], "ratio": 0.5, "mse": 0.5})
    estimates = pd.DataFrame(rows)

    fig = fig_split_strategy(estimates, estimates)
    left, right = fig.axes
    assert "MSE de validación" in left.get_ylabel() and "AP" not in left.get_ylabel()
    assert "10⁻³" in left.get_ylabel()
    # La mediana de cada caja es el MSE de la semilla del medio, en unidades de 1e-3.
    drawn = [np.asarray(line.get_ydata(), dtype=float) for line in left.lines]
    flat = [y[0] for y in drawn if y.size and np.ptp(y) == 0]  # medianas y topes de los bigotes
    assert pytest.approx(12.0) in flat and pytest.approx(12.1) in flat
    assert max(y.max() for y in drawn if y.size) < 100  # sin la fila de ratio 0.5
    assert "fraude" in right.get_ylabel()
    plt.close(fig)


# --- corrida completa ---


def _synthetic_csv(path: Path, n: int = 400) -> None:
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


def _small_config(name: str, tmp_path: Path) -> Path:
    config = json.loads((CONFIG_DIR / f"{name}.json").read_text())
    config["dataset"]["path"] = str(tmp_path / "fraud.csv")
    config["dataset"]["holdout_test"]["indices_path"] = str(tmp_path / "split/test_idx.npy")
    config["training"]["epochs"] = 5
    config["seeds"] = [0, 1]
    if name == "cv":
        config["sweep"] = {"training.optimizer.lr": [0.001, 0.01]}
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(config))
    return path


def _final_config(tmp_path: Path, results: Path, hash_: str) -> Path:
    config = json.loads(_small_config("cv", tmp_path).read_text())
    config["run_name"] = "ej1_final"
    config.pop("sweep")
    config["selected_from"] = str(results / "ej1_cv" / hash_)
    config["final"] = {"mode": "retrain"}
    path = tmp_path / "final.json"
    path.write_text(json.dumps(config))
    return path


def test_main_genera_todas_las_figuras(tmp_path):
    _synthetic_csv(tmp_path / "fraud.csv")
    results = tmp_path / "results"
    for name in ("cv", "split_strategy"):
        run_experiment(_small_config(name, tmp_path), results_dir=results, write=lambda _: None)
    run_best_fold(_small_config("best_fold", tmp_path), results_dir=results, write=lambda _: None)

    out = tmp_path / "figs"
    args = ["--results-dir", str(results), "--out-dir", str(out)]
    assert main(args) == 0
    for name in FIGURES:
        assert (out / f"{name}.png").exists(), name
        assert (out / f"{name}.pdf").exists(), name
    for name in FINAL_FIGURES:  # sin final_eval.json, el test no aparece
        assert not (out / f"{name}.png").exists(), name
    resumen = json.loads((out / "resumen.json").read_text())
    assert resumen["test"] is None
    assert resumen["recomendacion"]["threshold"] == pytest.approx(
        resumen["criterios"][resumen["recomendacion"]["criterio"]]["threshold"]
    )
    for key in ("tasa_fraude", "seleccion", "costos", "estabilidad", "split_strategy", "best_fold"):
        assert key in resumen, key

    # Con la evaluación final, aparecen el test y la figura de confusión.
    hash_ = resumen["seleccion"]["hash"]
    final_eval(_final_config(tmp_path, results, hash_), results_dir=results, write=lambda _: None)
    assert main(args) == 0
    for name in FINAL_FIGURES:
        assert (out / f"{name}.png").exists(), name
    resumen = json.loads((out / "resumen.json").read_text())
    test = resumen["test"]
    assert test["threshold"] == resumen["recomendacion"]["threshold"]
    assert test["n"] == len(np.load(tmp_path / "split/test_idx.npy"))
    assert test["bigmodel"]["average_precision"] == pytest.approx(1.0)
    cm = np.array(test["confusion_mean"])
    assert cm.sum() == pytest.approx(test["n"])


def test_main_sin_cv_devuelve_2(tmp_path):
    assert main(["--results-dir", str(tmp_path), "--out-dir", str(tmp_path / "o")]) == 2


def test_final_reentrena_exactamente_la_config_elegida_en_cv():
    final = load_config(CONFIG_DIR / "final.json", final_eval=True)
    assert final["run_name"] == "ej1_final"
    assert final["final"]["mode"] == "retrain"
    assert final["dataset"]["holdout_test"]["indices_path"] == TEST_IDX
    selected = Path(final["selected_from"])
    assert selected.parent == Path("results/ej1_cv")
    [run] = [
        r
        for r in expand(load_config(CONFIG_DIR / "cv.json"))
        if r["seed"] == 0 and r["fold"] == 0 and config_hash(r) == selected.name
    ]
    for key in ("dataset", "model", "training", "metrics"):
        assert final[key] == run[key], key
