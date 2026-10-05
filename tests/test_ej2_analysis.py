"""F12 · analysis/ej2.py: comparación de mecanismos de optimización (E2-02b).

Los resultados son sintéticos y viven en tmp_path: nunca se lee results/ ni se
escribe figures/ del repo, y el análisis no entrena (CLAUDE.md §2.6).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from analysis.ej2 import OPTIMIZER_VARIANTS, best_per_optimizer, plot_optimizers


def write_config(
    root: Path,
    experiment: str,
    hash8: str,
    training: dict,
    accuracies: list[float],
    epochs: list[int],
    *,
    statuses: list[str] | None = None,
    max_epochs: int = 300,
) -> None:
    """Una corrida por semilla de una config, con el config.json y metrics.json del runner."""
    statuses = statuses or ["ok"] * len(accuracies)
    for seed, (accuracy, trained, status) in enumerate(
        zip(accuracies, epochs, statuses, strict=True)
    ):
        run_dir = root / experiment / f"{hash8}_s{seed}"
        run_dir.mkdir(parents=True)
        config = {"run_name": experiment, "seed": seed, "training": training, "sweep": {}}
        stopped = trained < max_epochs
        metrics = {
            "hash": hash8,
            "seed": seed,
            "fold": None,
            "status": status,
            "epochs": max_epochs,
            "epochs_trained": trained,
            "best_epoch": trained - 20 if stopped else trained,
            "stopped_epoch": trained if stopped else None,
            "lr_final": training["optimizer"]["lr"],
            "val": {"loss": 1.0 - accuracy, "accuracy": accuracy, "macro_f1": accuracy - 0.1},
        }
        (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def training(kind: str, lr: float, adaptive_eta: dict | None = None, **params: float) -> dict:
    return {"optimizer": {"kind": kind, "lr": lr, **params}, "adaptive_eta": adaptive_eta}


ADAPTIVE = {"a": 0.01, "b": 0.1, "k": 5, "k_prime": 3, "monitor": "train_loss", "min_lr": 1e-8}


@pytest.fixture
def results_dir(tmp_path):
    root = tmp_path / "results"
    write_config(root, "ej2_lr", "gd000001", training("gd", 0.1), [0.90, 0.91, 0.92], [300] * 3)
    write_config(
        root, "ej2_lr", "gd000002", training("gd", 0.3), [0.94, 0.95, 0.96], [200, 210, 220]
    )
    write_config(
        root, "ej2_lr", "adam0001", training("adam", 0.001), [0.96, 0.97, 0.98], [60, 65, 70]
    )
    write_config(
        root, "ej2_lr", "adam0002", training("adam", 0.01), [0.93, 0.94, 0.95], [40, 45, 50]
    )
    momentum = [(0.01, [0.92, 0.93, 0.94]), (0.03, [0.95, 0.96, 0.97]), (0.1, [0.5, 0.6, 0.7])]
    for i, (lr, accuracies) in enumerate(momentum):
        write_config(
            root,
            "ej2_opt",
            f"mom0000{i}",
            training("momentum", lr, alpha=0.9),
            accuracies,
            [100, 110, 120],
        )
    for i, (lr, accuracy) in enumerate([(0.0001, 0.95), (0.0003, 0.96), (0.001, 0.94)]):
        write_config(
            root, "ej2_opt", f"rms0000{i}", training("rmsprop", lr), [accuracy] * 3, [80, 90, 100]
        )
    write_config(
        root,
        "ej2_eta_adapt",
        "eta00001",
        training("gd", 0.1, ADAPTIVE),
        [0.95, 0.95, 0.95],
        [150, 160, 170],
    )
    write_config(
        root,
        "ej2_eta_adapt",
        "eta00002",
        training("gd", 0.1, {**ADAPTIVE, "a": 0.05, "b": 0.5, "k": 3, "k_prime": 2}),
        [0.90, 0.91, 0.92],
        [120, 130, 140],
    )
    return root


def test_best_per_optimizer_elige_el_mejor_eta_de_cada_mecanismo(results_dir):
    best = best_per_optimizer(results_dir)
    # Orden fijo: el de OPTIMIZER_VARIANTS, no el del ranking.
    assert list(best["label"]) == [label for label, *_ in OPTIMIZER_VARIANTS]
    assert list(best["label"]) == ["SGD", "Momentum", "RMSProp", "Adam", "SGD + η adaptativo"]
    assert list(best["hash"]) == ["gd000002", "mom00001", "rms00001", "adam0001", "eta00001"]
    np.testing.assert_allclose(best["lr"], [0.3, 0.03, 0.0003, 0.001, 0.1])
    assert list(best["n_configs"]) == [2, 3, 3, 2, 2]
    assert list(best["n"]) == [3] * 5

    sgd = best.set_index("label").loc["SGD"]
    np.testing.assert_allclose(sgd["val.accuracy_mean"], 0.95)
    np.testing.assert_allclose(sgd["val.accuracy_std"], np.std([0.94, 0.95, 0.96], ddof=1))
    np.testing.assert_allclose(sgd["epochs_trained_mean"], 210.0)
    np.testing.assert_allclose(sgd["epochs_trained_std"], 10.0)
    assert sgd["n_stopped"] == 3


def test_best_per_optimizer_guarda_los_hiperparametros_de_cada_mecanismo(results_dir):
    best = best_per_optimizer(results_dir).set_index("label")
    assert best.loc["Momentum", "alpha"] == 0.9
    assert np.isnan(best.loc["Adam", "alpha"])
    adaptive = best.loc["SGD + η adaptativo"]
    assert (adaptive["a"], adaptive["b"], adaptive["k"], adaptive["k_prime"]) == (0.01, 0.1, 5, 3)
    assert np.isnan(best.loc["SGD", "a"])


def test_best_per_optimizer_no_mezcla_gd_fijo_con_gd_adaptativo(results_dir):
    # Un GD con η adaptativo dentro de ej2_lr no compite como "SGD" de η fijo.
    write_config(
        results_dir, "ej2_lr", "gdadapt1", training("gd", 0.3, ADAPTIVE), [0.99] * 3, [50] * 3
    )
    best = best_per_optimizer(results_dir).set_index("label")
    assert best.loc["SGD", "hash"] == "gd000002"


def test_best_per_optimizer_descarta_configs_con_corridas_divergidas(results_dir):
    write_config(
        results_dir,
        "ej2_lr",
        "adam0003",
        training("adam", 0.003),
        [0.99, 0.99, 0.99],
        [30, 30, 21],
        statuses=["ok", "ok", "diverged"],
    )
    best = best_per_optimizer(results_dir).set_index("label")
    assert best.loc["Adam", "hash"] == "adam0001"
    assert best.loc["Adam", "n_configs"] == 3


def test_best_per_optimizer_cuenta_las_corridas_que_no_cortaron(results_dir):
    write_config(
        results_dir, "ej2_lr", "gd000003", training("gd", 1.0), [0.97] * 3, [300, 300, 250]
    )
    sgd = best_per_optimizer(results_dir).set_index("label").loc["SGD"]
    assert sgd["hash"] == "gd000003" and sgd["n_stopped"] == 1


def test_best_per_optimizer_sin_resultados_es_un_error_claro(results_dir, tmp_path):
    with pytest.raises(FileNotFoundError, match="ej2_lr"):
        best_per_optimizer(tmp_path / "vacio")


def test_plot_optimizers_escribe_png_y_pdf(results_dir, tmp_path):
    out_dir = tmp_path / "figures"
    paths = plot_optimizers(results_dir, out_dir)
    assert [p.name for p in paths] == ["E2-02b_optimizers.png", "E2-02b_optimizers.pdf"]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
