"""analysis/common.py y analysis/aggregate.py (F07): solo leen results/, nunca entrenan."""

import json

import numpy as np
import pandas as pd
import pytest

from analysis.aggregate import main as aggregate_main
from analysis.aggregate import summarize
from analysis.common import (
    PALETTE,
    aggregate,
    apply_style,
    flatten,
    load_histories,
    load_runs,
    save_figure,
)
from experiments.runner import run_experiment


def test_aggregate_media_desvio_min_max_y_n():
    df = pd.DataFrame(
        {
            "hash": ["a", "a", "a", "b", "b"],
            "seed": [0, 1, 2, 0, 1],
            "val.accuracy": [0.8, 0.9, 1.0, 0.5, 0.7],
            "time_total_s": [1.0, 2.0, 3.0, 4.0, 4.0],
        }
    )
    out = aggregate(df, "hash").set_index("hash")
    assert list(out["n"]) == [3, 2]
    np.testing.assert_allclose(out["val.accuracy_mean"], [0.9, 0.6])
    # Desvío muestral (ddof = 1), el que se reporta como media ± desvío entre semillas.
    np.testing.assert_allclose(
        out["val.accuracy_std"], [np.std([0.8, 0.9, 1.0], ddof=1), np.std([0.5, 0.7], ddof=1)]
    )
    np.testing.assert_allclose(out["val.accuracy_min"], [0.8, 0.5])
    np.testing.assert_allclose(out["val.accuracy_max"], [1.0, 0.7])
    np.testing.assert_allclose(out["time_total_s_std"], [1.0, 0.0])
    assert "seed_mean" not in out.columns  # la semilla no es una métrica


def test_aggregate_por_varias_claves_y_metricas_elegidas():
    df = pd.DataFrame({"lr": [0.1, 0.1, 0.2], "opt": ["gd", "gd", "gd"], "m": [1.0, 3.0, 5.0]})
    out = aggregate(df, ["lr", "opt"], metrics=["m"])
    assert list(out.columns) == ["lr", "opt", "n", "m_mean", "m_std", "m_min", "m_max"]
    assert list(out["m_mean"]) == [2.0, 5.0] and np.isnan(out["m_std"].iloc[1])


def test_flatten_config_y_metricas():
    nested = {"a": {"b": 1, "c": [1, 2]}, "d": [[1, 2], [3, 4]], "e": ["x"], "f": None}
    assert flatten(nested) == {
        "a.b": 1,
        "a.c": "[1, 2]",
        "d": "[[1, 2], [3, 4]]",
        "e": '["x"]',
        "f": None,
    }
    assert flatten(nested, expand_lists=True) == {
        "a.b": 1,
        "a.c.0": 1,
        "a.c.1": 2,
        "d": "[[1, 2], [3, 4]]",
        "e": '["x"]',
        "f": None,
    }


@pytest.fixture
def sweep_dir(tmp_path):
    config = {
        "run_name": "xor",
        "seeds": [0, 1],
        "dataset": {"synthetic": {"name": "xor"}, "normalize": "none", "split": {"kind": "none"}},
        "model": {"layers": [2, 3, 1], "output_activation": "tanh"},
        "training": {"optimizer": {"kind": "gd", "lr": 0.1}, "epochs": 15},
        "metrics": ["accuracy"],
        "sweep": {"training.optimizer.lr": [0.05, 0.2]},
    }
    path = tmp_path / "xor.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    run_experiment(path, results_dir=tmp_path, write=lambda line: None)
    return tmp_path / "xor"


def test_load_runs_una_fila_por_corrida(sweep_dir):
    runs = load_runs(sweep_dir)
    assert len(runs) == 4
    assert set(runs["training.optimizer.lr"]) == {0.05, 0.2}
    assert runs["model.layers"].iloc[0] == "[2, 3, 1]"
    assert {"run", "hash", "seed", "status", "train.loss", "train.accuracy"} <= set(runs.columns)
    assert "train.per_class.recall.1" in runs.columns
    metrics = json.loads((sweep_dir / runs["run"].iloc[0] / "metrics.json").read_text("utf-8"))
    assert runs["train.loss"].iloc[0] == metrics["train"]["loss"]


def test_load_histories_formato_largo(sweep_dir):
    histories = load_histories(sweep_dir)
    assert list(histories.columns) == ["run", "hash", "seed", "fold", "epoch", "metric", "value"]
    one = histories[(histories["run"] == histories["run"].iloc[0])]
    assert sorted(one.loc[one["metric"] == "train_loss", "epoch"]) == list(range(1, 16))
    assert "val_loss" not in set(histories["metric"])  # sin validación: no aparece vacía
    assert histories["value"].notna().all()


def test_summary_csv_una_fila_por_hash(sweep_dir, capsys):
    assert aggregate_main([str(sweep_dir)]) == 0
    summary = pd.read_csv(sweep_dir / "summary.csv")
    assert len(summary) == 2 and list(summary["n"]) == [2, 2]
    assert list(summary.columns[:5]) == ["hash", "training.optimizer.lr", "n", "n_ok", "n_diverged"]
    runs = load_runs(sweep_dir)
    for _, row in summary.iterrows():
        group = runs[runs["hash"] == row["hash"]]
        assert row["train.loss_mean"] == pytest.approx(group["train.loss"].mean())
        assert row["train.loss_std"] == pytest.approx(group["train.loss"].std(ddof=1))
    assert summarize(sweep_dir).shape == summary.shape
    assert "summary.csv" in capsys.readouterr().out


def test_aggregate_sin_corridas_es_error(tmp_path):
    assert aggregate_main([str(tmp_path)]) == 2


def test_estilo_y_guardado_png_y_pdf(tmp_path):
    import matplotlib.pyplot as plt

    apply_style()
    assert plt.rcParams["font.size"] >= 14 and plt.rcParams["axes.labelsize"] >= 14
    assert plt.rcParams["xtick.labelsize"] >= 14 and plt.rcParams["legend.fontsize"] >= 14
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    assert [c.lower() for c in colors] == [c.lower() for c in PALETTE]

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9], label="serie")
    paths = save_figure(fig, tmp_path / "figs" / "E0-prueba")
    assert [p.name for p in paths] == ["E0-prueba.png", "E0-prueba.pdf"]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
    assert not plt.fignum_exists(fig.number)  # save_figure cierra la figura
