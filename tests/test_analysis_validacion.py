"""Figuras de validación (F08-T5 y T6): helpers de analysis/validacion.py y corrida completa.

La corrida completa usa las configs reales de experiments/configs/validacion/
con menos semillas y épocas, en tmp_path: nunca toca results/ ni figures/ del repo.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.validacion import (
    FIGURES,
    VALIDATION_CONFIGS,
    convergence_epoch,
    curve_stats,
    decision_line,
    init_label,
    init_runs,
    main,
)
from experiments.runner import run_experiment

CONFIG_DIR = Path(__file__).parents[1] / "experiments/configs/validacion"


# --- decision_line ---


def test_recta_de_decision_general():
    # x1 + 2 x2 − 1 = 0  →  x2 = (1 − x1) / 2
    xs, ys = decision_line(np.array([1.0, 2.0]), -1.0, lim=2.0)
    np.testing.assert_allclose(xs, [-2.0, 2.0])
    np.testing.assert_allclose(ys, [1.5, -0.5])


def test_recta_de_decision_vertical_y_horizontal():
    xs, ys = decision_line(np.array([2.0, 0.0]), -1.0, lim=1.5)  # x1 = 0.5
    np.testing.assert_allclose(xs, [0.5, 0.5])
    np.testing.assert_allclose(ys, [-1.5, 1.5])
    xs, ys = decision_line(np.array([0.0, -4.0]), 2.0, lim=1.0)  # x2 = 0.5
    np.testing.assert_allclose(xs, [-1.0, 1.0])
    np.testing.assert_allclose(ys, [0.5, 0.5])


def test_sin_pesos_no_hay_recta():
    assert decision_line(np.zeros(2), 0.3, lim=1.0) is None


# --- convergence_epoch ---


@pytest.mark.parametrize(
    "errors, expected",
    [
        ([0.5, 0.25, 0.0, 0.0], 3),
        ([0.5, 0.0, 0.25, 0.0], 4),  # vuelve a fallar: cuenta la última entrada a 0
        ([0.0, 0.0], 1),
        ([0.5, 0.25], None),
        ([0.25, 0.0, 0.25], None),
    ],
)
def test_epoca_de_convergencia(errors, expected):
    epochs = np.arange(1, len(errors) + 1)
    assert convergence_epoch(epochs, np.array(errors)) == expected


# --- curve_stats ---


def test_curve_stats_media_desvio_y_n_por_epoca():
    histories = pd.DataFrame(
        {
            "run": ["a", "a", "b", "b", "a"],
            "epoch": [1, 2, 1, 2, 1],
            "metric": ["train_accuracy"] * 4 + ["train_loss"],
            "value": [0.5, 1.0, 0.25, 0.5, 9.0],
        }
    )
    stats = curve_stats(histories, "train_accuracy", transform=lambda v: 1 - v)
    assert list(stats.columns) == ["epoch", "mean", "std", "n"]
    np.testing.assert_allclose(stats["mean"], [0.625, 0.25])
    expected_std = [np.std([0.5, 0.75], ddof=1), np.std([0.0, 0.5], ddof=1)]
    np.testing.assert_allclose(stats["std"], expected_std)
    assert list(stats["n"]) == [2, 2]


def test_curve_stats_sin_la_metrica_es_error():
    histories = pd.DataFrame({"run": ["a"], "epoch": [1], "metric": ["train_loss"], "value": [1]})
    with pytest.raises(ValueError, match="val_loss"):
        curve_stats(histories, "val_loss")


# --- init_label ---


def test_etiqueta_de_inicializacion():
    assert init_label("uniform", {"low": -0.1, "high": 0.1}) == "U(±0.1)"
    assert init_label("uniform", {"low": -1.0, "high": 1.0}) == "U(±1)"
    assert init_label("uniform", {}) == "U(±0.5)"  # default de Uniform
    assert init_label("xavier", {}) == "Xavier"
    assert init_label("uniform", {"low": 0.0, "high": 1.0}) == "U(0, 1)"


# --- Corrida completa ---

# Semillas y épocas reducidas: alcanza para que existan todas las figuras.
SHORT = {"and_step": 15, "linear_identity": 40, "nonlinear_tanh": 40}
SHORT_XOR = 150


@pytest.fixture(scope="module")
def validation_results(tmp_path_factory):
    root = tmp_path_factory.mktemp("validacion")
    for name in VALIDATION_CONFIGS:
        config = json.loads((CONFIG_DIR / f"{name}.json").read_text(encoding="utf-8"))
        config["seeds"] = [0, 1, 2]
        config["training"]["epochs"] = SHORT.get(name, SHORT_XOR)
        path = root / f"{name}.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        results = run_experiment(path, results_dir=root / "results", write=lambda line: None)
        assert {r["status"] for r in results} == {"ok"}, name
    return root


def test_las_configs_de_validacion_son_las_del_repo():
    assert sorted(VALIDATION_CONFIGS) == sorted(p.stem for p in CONFIG_DIR.glob("*.json"))


def test_genera_todas_las_figuras_y_el_resumen(validation_results, capsys):
    out = validation_results / "figs"
    assert main(["--results-dir", str(validation_results / "results"), "--out-dir", str(out)]) == 0

    for name in FIGURES:
        assert name.startswith("V-0"), name
        for ext in (".png", ".pdf"):
            path = out / (name + ext)
            assert path.exists() and path.stat().st_size > 0, path.name

    summary = json.loads((out / "resumen.json").read_text(encoding="utf-8"))
    assert summary["V-01"]["n_seeds"] == 3
    assert summary["V-01"]["converged"] == 3  # AND converge con todas las semillas
    assert summary["V-02"]["train_mse"]["mean"] < 1e-2
    assert summary["V-04"]["step"]["min_error"] >= 0.25  # una recta no separa XOR
    init = summary["V-04_init"]["by_init"]
    assert set(init) == {"U(±0.1)", "U(±0.5)", "U(±1)", "Xavier"}
    for label, by_arch in init.items():
        assert set(by_arch) == {"[2, 2, 1]", "[2, 3, 2, 1]"}, label
        for group in by_arch.values():
            assert group["n"] == 3 and 0 <= group["solved"] <= 3
    assert "V-01" in capsys.readouterr().out


def test_t6_junta_uniformes_y_xavier(validation_results):
    runs = init_runs(validation_results / "results")
    # 3 escalas uniformes + Xavier, 2 arquitecturas, 3 semillas.
    assert len(runs) == 4 * 2 * 3
    assert set(runs["init"]) == {"U(±0.1)", "U(±0.5)", "U(±1)", "Xavier"}
    assert set(runs["model.layers"]) == {"[2, 2, 1]", "[2, 3, 2, 1]"}
    assert set(runs.loc[runs["init"] == "Xavier", "run_name"]) == {"val_xor_221", "val_xor_2321"}


def test_t6_exige_que_xavier_tenga_la_misma_config_que_las_uniformes(validation_results, tmp_path):
    # Xavier sale de val_xor_221 y val_xor_2321: si no se corrieron con las
    # mismas épocas que val_xor_init_uniform, la comparación no es justa.
    results, fake = validation_results / "results", tmp_path / "results"
    for name in ("val_xor_init_uniform", "val_xor_2321"):
        shutil.copytree(results / name, fake / name)
    config = json.loads((CONFIG_DIR / "xor_221.json").read_text(encoding="utf-8"))
    config["seeds"] = [0, 1, 2]
    config["training"]["epochs"] = SHORT_XOR + 10
    path = tmp_path / "xor_221.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    run_experiment(path, results_dir=fake, write=lambda line: None)
    with pytest.raises(ValueError, match="training.epochs"):
        init_runs(fake)


def test_sin_resultados_avisa_que_hay_que_correr_el_runner(tmp_path, capsys):
    assert main(["--results-dir", str(tmp_path), "--out-dir", str(tmp_path / "figs")]) == 2
    assert "experiments.runner" in capsys.readouterr().err
