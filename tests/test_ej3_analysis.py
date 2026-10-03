"""F13 · analysis/ej3.py: grilla de ablación (E3-02) y descomposición de la mejora en test.

Los resultados son sintéticos y viven en tmp_path: nunca se lee results/ ni
digits_test.csv, ni se escribe figures/ del repo (CLAUDE.md §2.6).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.ej3 import (
    ABLATION_NETWORKS,
    ABLATION_SHIFTS,
    ablation_grid,
    decompose_test_gain,
    plot_ablation_ranking,
    plot_accuracy_progression,
)

SUPPORT = [10, 20, 30, 40]


def search_rows() -> list[dict]:
    """Una fila de summary.csv por celda de la grilla, más una red que no entra en ella."""
    networks = [(activation, sizes) for _, activation, sizes in ABLATION_NETWORKS]
    networks.append(("relu", [784, 128, 64, 32, 10]))
    rows = []
    for i, (activation, sizes) in enumerate(networks):
        for j, (_, max_px) in enumerate(ABLATION_SHIFTS):
            rows.append(
                {
                    "hash": f"h{i}{j}",
                    "model.layers": json.dumps(sizes),
                    "model.hidden_activation": activation,
                    "training.augmentation.kind": None if max_px is None else "random_shift",
                    "training.augmentation.max_px": max_px,
                    "n": 3,
                    "val.accuracy_mean": 0.90 + 0.01 * i + 0.001 * j,
                    "val.accuracy_std": 0.001 * (i + 1),
                }
            )
    return rows


def write_summary(root: Path, experiment: str, rows: list[dict]) -> None:
    (root / experiment).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / experiment / "summary.csv", index=False)


def write_final_eval(root: Path, experiment: str, recalls: list[list[float]]) -> None:
    """final_eval.json con un modelo por fila de recalls (una columna por clase)."""
    models = []
    for recall in recalls:
        accuracy = float(np.dot(SUPPORT, recall) / sum(SUPPORT))
        models.append(
            {"test": {"accuracy": accuracy, "per_class": {"recall": recall, "support": SUPPORT}}}
        )
    accuracies = [m["test"]["accuracy"] for m in models]
    report = {
        "models": models,
        "mean": {"accuracy": float(np.mean(accuracies))},
        "std": {"accuracy": float(np.std(accuracies, ddof=1))},
    }
    (root / experiment).mkdir(parents=True, exist_ok=True)
    (root / experiment / "final_eval.json").write_text(json.dumps(report), encoding="utf-8")


@pytest.fixture
def results_dir(tmp_path):
    root = tmp_path / "results"
    rows = search_rows()
    write_summary(root, "ej3_search", rows)
    write_summary(root, "ej3_baseline", rows[:1])
    write_summary(root, "ej3_best", rows[-1:])
    # La clase 2 no existe en el entrenamiento de "ej2": recall 0 en todos los modelos.
    write_final_eval(root, "ej2_final", [[1.0, 0.9, 0.0, 0.8], [1.0, 0.7, 0.0, 0.8]])
    write_final_eval(root, "ej3_final", [[1.0, 0.9, 0.9, 0.9], [0.9, 0.9, 0.7, 0.9]])
    return root


# --- grilla de ablación ---


def test_ablation_grid_lee_las_nueve_celdas_del_summary(results_dir):
    grid = ablation_grid(results_dir)
    networks = [name for name, *_ in ABLATION_NETWORKS]
    shifts = [name for name, _ in ABLATION_SHIFTS]
    assert networks == ["tanh [128]", "ReLU [128]", "ReLU [256, 128]"]
    assert shifts == ["Sin shift", "Shift ±1 px", "Shift ±2 px"]
    # Orden fijo: redes por filas, shift por columnas; la cuarta red del barrido no entra.
    assert list(grid["network"]) == [n for n in networks for _ in shifts]
    assert list(grid["shift"]) == shifts * 3
    assert list(grid["hash"]) == [f"h{i}{j}" for i in range(3) for j in range(3)]
    np.testing.assert_allclose(
        grid["accuracy_mean"], [0.90 + 0.01 * i + 0.001 * j for i in range(3) for j in range(3)]
    )
    np.testing.assert_allclose(
        grid["accuracy_std"], [0.001 * (i + 1) for i in range(3) for _ in range(3)]
    )
    assert set(grid["n"]) == {3}


def test_ablation_grid_exige_cada_celda_una_sola_vez(results_dir):
    rows = search_rows()
    write_summary(results_dir, "ej3_search", rows[1:])
    with pytest.raises(ValueError, match="tanh"):
        ablation_grid(results_dir)
    write_summary(results_dir, "ej3_search", [*rows, {**rows[4], "hash": "repetida"}])
    with pytest.raises(ValueError, match="ReLU"):
        ablation_grid(results_dir)


def test_plot_ablation_ranking_escribe_png_y_pdf(results_dir, tmp_path):
    paths = plot_ablation_ranking(results_dir, tmp_path / "figures")
    assert [p.name for p in paths] == ["E3-02_ablation_ranking.png", "E3-02_ablation_ranking.pdf"]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


# --- descomposición de la mejora en test ---


def test_decompose_test_gain_suma_la_diferencia_de_accuracy(results_dir):
    gain = decompose_test_gain(results_dir)
    assert list(gain["digit"]) == [0, 1, 2, 3]
    assert list(gain["n"]) == SUPPORT
    np.testing.assert_allclose(gain["recall_ej2"], [1.0, 0.8, 0.0, 0.8])
    np.testing.assert_allclose(gain["recall_ej3"], [0.95, 0.9, 0.8, 0.9])
    # Aporte de cada clase en puntos de accuracy: (n_c / N) · Δrecall_c · 100.
    np.testing.assert_allclose(gain["contribution"], [-0.5, 2.0, 24.0, 4.0])
    ej2 = json.loads((results_dir / "ej2_final" / "final_eval.json").read_text(encoding="utf-8"))
    ej3 = json.loads((results_dir / "ej3_final" / "final_eval.json").read_text(encoding="utf-8"))
    total = (ej3["mean"]["accuracy"] - ej2["mean"]["accuracy"]) * 100.0
    np.testing.assert_allclose(gain["contribution"].sum(), total)


def test_decompose_test_gain_exige_el_mismo_test(results_dir):
    report = json.loads((results_dir / "ej3_final" / "final_eval.json").read_text(encoding="utf-8"))
    report["models"][0]["test"]["per_class"]["support"] = [10, 20, 30, 41]
    (results_dir / "ej3_final" / "final_eval.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="mismo test"):
        decompose_test_gain(results_dir)


# --- progresión ---


def test_plot_accuracy_progression_escribe_png_y_pdf(results_dir, tmp_path):
    paths = plot_accuracy_progression(results_dir, tmp_path / "figures")
    assert [p.name for p in paths] == [
        "E3-01_accuracy_progression.png",
        "E3-01_accuracy_progression.pdf",
    ]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
