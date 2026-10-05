"""F12 · Tests de configuración y pipeline de experimentos de dígitos (Ej2).

Verifica que base.json cumpla el contrato de configuración
(CLAUDE.md §6, docs/03-arquitectura.md §4), respete las restricciones
del enunciado (R-02: test sagrado), y que el runner pueda entrenar
un modelo sobre formato 'digits' sin errores.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.loaders import DIGITS_N_PIXELS
from experiments.config import expand, load_config, run_id
from experiments.runner import run_experiment

ROOT = Path(__file__).parents[1]
CONFIG_DIR = ROOT / "experiments/configs/ej2"
BASE_CONFIG_PATH = CONFIG_DIR / "base.json"


def synthetic_digits_csv(path: Path, n_samples: int = 40, seed: int = 0) -> None:
    """Crea un archivo CSV sintético válido con formato 'digits'."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 10, size=n_samples)
    images = [
        json.dumps(np.round(rng.uniform(0.0, 1.0, DIGITS_N_PIXELS), 4).tolist())
        for _ in range(n_samples)
    ]
    pd.DataFrame({"label": labels, "image": images}).to_csv(path, index=False)


# --- Tests de la configuración base ---


def test_base_config_existe_y_valida():
    assert BASE_CONFIG_PATH.exists(), f"No existe {BASE_CONFIG_PATH}"
    config = load_config(BASE_CONFIG_PATH)
    assert config["run_name"] == "ej2_base"


def test_base_config_respeta_disciplina_de_datos_sagrados():
    config = load_config(BASE_CONFIG_PATH)
    ds = config["dataset"]

    # R-02 y C5: digits_test.csv no se toca durante desarrollo
    assert "digits_test.csv" not in str(ds["path"])
    assert ds["holdout_test"] is None
    assert ds["test_path"] is None


def test_base_config_arquitectura_y_codificacion():
    config = load_config(BASE_CONFIG_PATH)
    ds = config["dataset"]
    model = config["model"]

    assert ds["format"] == "digits"
    assert ds["normalize"] == "none"
    assert ds["target_encoding"] == "onehot"
    assert ds["n_classes"] == 10

    # Red feedforward de entrada 784 y salida 10
    assert model["layers"][0] in (784, "auto")
    assert model["layers"][-1] == 10
    assert model["hidden_activation"] == "tanh"
    assert model["output_activation"] == "sigmoid"


def test_base_config_optimizador_y_reproducibilidad():
    config = load_config(BASE_CONFIG_PATH)
    training = config["training"]

    # C4: mínimo 3 semillas
    assert len(config["seeds"]) >= 3

    # Default de la cátedra: SGD con minibatch 32 y loss MSE
    assert training["optimizer"]["kind"] == "gd"
    assert training["batch_size"] == 32
    assert training["loss"] == "mse"

    # Early stopping configurado
    assert training["early_stopping"] is not None
    assert training["early_stopping"]["monitor"] == "val_loss"
    assert training["early_stopping"]["mode"] == "min"

    # Métricas requeridas para clasificación multiclase
    assert "accuracy" in config["metrics"]
    assert "macro_f1" in config["metrics"]


def test_lr_config_existe_y_valida_sweep():
    lr_config_path = ROOT / "experiments/configs/ej2/lr.json"
    assert lr_config_path.exists(), f"No existe {lr_config_path}"
    config = load_config(lr_config_path)
    assert config["run_name"] == "ej2_lr"
    assert "training.optimizer.kind" in config["sweep"]
    assert "training.optimizer.lr" in config["sweep"]
    assert set(config["sweep"]["training.optimizer.kind"]) == {"gd", "adam"}


def _sin_barrido(config: dict) -> dict:
    """La config sin lo que cambia entre experimentos del estudio OFAT."""
    out = {k: v for k, v in config.items() if k not in ("run_name", "sweep")}
    out["training"] = {
        k: v for k, v in config["training"].items() if k not in ("optimizer", "adaptive_eta")
    }
    return out


@pytest.mark.parametrize("name", ["opt", "eta_adapt"])
def test_configs_de_optimizacion_solo_cambian_el_optimizador(name):
    lr = load_config(CONFIG_DIR / "lr.json")
    config = load_config(CONFIG_DIR / f"{name}.json")
    assert config["run_name"] == f"ej2_{name}"
    # Misma red, datos, semillas y early stopping que ej2_lr: comparables entre sí.
    assert _sin_barrido(config) == _sin_barrido(lr)
    ds = config["dataset"]
    assert "digits_test.csv" not in str(ds["path"])
    assert ds["holdout_test"] is None and ds["test_path"] is None


def test_opt_config_mini_barrido_de_tres_eta_por_optimizador():
    config = load_config(CONFIG_DIR / "opt.json")
    assert list(config["sweep"]) == ["training.optimizer"]
    optimizers = config["sweep"]["training.optimizer"]
    by_kind: dict[str, list[dict]] = {}
    for optimizer in optimizers:
        by_kind.setdefault(optimizer["kind"], []).append(optimizer)
    assert set(by_kind) == {"momentum", "rmsprop"}
    assert sorted(o["lr"] for o in by_kind["momentum"]) == [0.01, 0.03, 0.1]
    assert sorted(o["lr"] for o in by_kind["rmsprop"]) == [0.0001, 0.0003, 0.001]
    assert all(o["alpha"] == 0.9 for o in by_kind["momentum"])
    runs = expand(config)
    assert len(runs) == len(optimizers) * len(config["seeds"]) == 18
    assert len({run_id(r) for r in runs}) == 18


def test_eta_adapt_config_barre_el_eta_adaptativo_sobre_gd():
    config = load_config(CONFIG_DIR / "eta_adapt.json")
    assert config["training"]["optimizer"] == {"kind": "gd", "lr": 0.1}
    assert list(config["sweep"]) == ["training.adaptive_eta"]
    combos = config["sweep"]["training.adaptive_eta"]
    assert combos == [
        {"a": 0.01, "b": 0.1, "k": 5, "k_prime": 3},
        {"a": 0.05, "b": 0.5, "k": 3, "k_prime": 2},
    ]
    runs = expand(config)
    assert len(runs) == len(combos) * len(config["seeds"]) == 6
    assert all(r["training"]["adaptive_eta"]["monitor"] == "train_loss" for r in runs)


# --- Smoke test del runner con formato digits en tmp_path ---


def test_runner_entrena_dataset_digits_en_tmp_path(tmp_path):
    csv_file = tmp_path / "digits_synthetic.csv"
    synthetic_digits_csv(csv_file, n_samples=30, seed=42)

    # Configuración de prueba rápida: 1 semilla, 2 épocas, red chica
    cfg = {
        "run_name": "ej2_smoke_test",
        "seeds": [0],
        "dataset": {
            "format": "digits",
            "path": str(csv_file),
            "normalize": "none",
            "target_encoding": "onehot",
            "n_classes": 10,
            "split": {"kind": "holdout", "ratio": 0.8, "stratified": True},
        },
        "model": {"layers": [784, 16, 10]},
        "training": {
            "loss": "mse",
            "optimizer": {"kind": "gd", "lr": 0.05},
            "batch_size": 16,
            "epochs": 2,
        },
        "metrics": ["accuracy", "macro_f1"],
        "logging": {"every": 1, "save_model": True, "save_predictions": True},
    }

    cfg_path = tmp_path / "smoke_config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    results_dir = tmp_path / "results"
    runs = run_experiment(cfg_path, results_dir=results_dir)

    assert len(runs) == 1
    assert runs[0]["status"] == "ok"

    run_dir = Path(runs[0]["run_dir"])
    assert (run_dir / "config.json").exists()
    assert (run_dir / "history.csv").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "predictions.npz").exists()
    assert (run_dir / "model.npz").exists()

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "train" in metrics and "val" in metrics
    assert "accuracy" in metrics["val"]
    assert "macro_f1" in metrics["val"]
    assert metrics["status"] == "ok"
