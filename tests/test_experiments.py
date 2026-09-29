"""Infraestructura de experimentos (F07): configs, runner, reanudación y agregación.

Todo corre en tmp_path con datasets sintéticos o CSV chicos escritos por el
test: nunca toca results/ ni datasets/ del repo.
"""

import copy
import json
import time
from pathlib import Path

import numpy as np
import pytest

from core.augmentation import Compose, GaussianNoise, RandomShift
from core.callbacks import AdaptiveEta, EarlyStopping
from core.optimizers import Adam
from core.serialization import load_checkpoint
from experiments.config import (
    ConfigError,
    build,
    config_hash,
    expand,
    load_config,
    resolve_config,
    run_id,
)
from experiments.runner import (
    RunnerError,
    final_eval,
    main,
    resume_run,
    run_experiment,
    train_run,
)


def xor_config(**sections) -> dict:
    """Config chica de XOR; cada kwarg reemplaza una sección de primer nivel."""
    config = {
        "run_name": "xor",
        "seeds": [0],
        "dataset": {"synthetic": {"name": "xor"}, "normalize": "none", "split": {"kind": "none"}},
        "model": {"layers": [2, 3, 1], "output_activation": "tanh"},
        "training": {"optimizer": {"kind": "gd", "lr": 0.1}, "epochs": 20},
        "metrics": ["accuracy"],
        "logging": {"every": 10},
    }
    config.update(sections)
    return config


def with_key(config: dict, key: str, value) -> dict:
    """Copia de config con la clave en notación punto reemplazada."""
    config = copy.deepcopy(config)
    node = config
    *parents, last = key.split(".")
    for part in parents:
        node = node.setdefault(part, {})
    node[last] = value
    return config


def write_json(path, data) -> str:
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


# --- load_config: defaults ---


def test_config_minima_se_completa_con_defaults(tmp_path):
    minimal = {
        "run_name": "mini",
        "dataset": {"synthetic": {"name": "xor"}},
        "model": {"layers": [2, 2, 1]},
        "training": {"epochs": 5},
    }
    config = load_config(write_json(tmp_path / "c.json", minimal))

    assert config["seeds"] == [0, 1, 2]
    assert config["trainer"] == "backprop"
    dataset = config["dataset"]
    assert dataset["format"] == "tabular" and dataset["normalize"] == "minmax"
    assert dataset["target_encoding"] == "none" and dataset["task"] == "auto"
    assert dataset["split"] == {"kind": "holdout", "ratio": 0.8, "stratified": True, "seed": None}
    assert dataset["holdout_test"] is None and dataset["test_path"] is None
    model = config["model"]
    assert (model["hidden_activation"], model["output_activation"]) == ("tanh", "identity")
    assert (model["beta"], model["initializer"], model["init_params"]) == (1.0, "xavier", {})
    training = config["training"]
    assert training["loss"] == "mse" and training["optimizer"] == {"kind": "gd", "lr": 0.01}
    assert training["batch_size"] is None and training["l2"] == 0.0
    assert training["early_stopping"] is None and training["adaptive_eta"] is None
    assert config["metrics"] == [] and config["sweep"] == {}
    assert config["logging"] == {
        "every": 10,
        "save_model": True,
        "save_predictions": False,
        "checkpoint_every": 50,
    }


def test_seed_suelto_es_una_lista_de_una_semilla():
    config = xor_config()
    del config["seeds"]
    config["seed"] = 7
    assert resolve_config(config)["seeds"] == [7]


def test_lr_por_defecto_es_el_del_optimizador():
    config = resolve_config(with_key(xor_config(), "training.optimizer", {"kind": "adam"}))
    assert config["training"]["optimizer"] == {"kind": "adam", "lr": 0.001}


def test_defaults_de_early_stopping_y_eta_adaptativo():
    config = xor_config(dataset={"synthetic": {"name": "xor"}, "normalize": "none"})
    config["training"]["early_stopping"] = {"patience": 5}
    config["training"]["adaptive_eta"] = {"a": 0.01, "b": 0.5, "k": 3, "k_prime": 2}
    training = resolve_config(config)["training"]
    assert training["early_stopping"] == {
        "patience": 5,
        "monitor": "val_loss",
        "mode": "min",
        "min_delta": 0.0,
        "restore_best": True,
    }
    assert training["adaptive_eta"]["monitor"] == "train_loss"


def test_resolver_dos_veces_da_lo_mismo():
    config = xor_config()
    config["training"]["augmentation"] = {"kind": "gaussian_noise", "sigma": 0.1}
    once = resolve_config(config)
    assert resolve_config(once) == once


def test_random_shift_en_digitos_completa_image_shape():
    config = xor_config(
        dataset={"format": "digits", "path": "d.csv", "target_encoding": "onehot"},
        model={"layers": [784, 10]},
        metrics=[],
    )
    config["training"]["augmentation"] = [
        {"kind": "random_shift", "max_px": 2},
        {"kind": "gaussian_noise", "sigma": 0.05, "clip": [0, 1]},
    ]
    augmentation = resolve_config(config)["training"]["augmentation"]
    assert augmentation[0]["image_shape"] == [28, 28]


# --- Validación ---


INVALID = {
    "clave_desconocida_arriba": (lambda c: with_key(c, "optimiser", {}), "optimiser"),
    "clave_desconocida_anidada": (
        lambda c: with_key(c, "dataset.split.ratio", 0.5),
        "dataset.split.ratio",
    ),
    "parametro_de_otro_optimizador": (
        lambda c: with_key(c, "training.optimizer.alpha", 0.9),
        "training.optimizer.alpha",
    ),
    "softmax_con_mse": (
        lambda c: {
            **c,
            "dataset": {"format": "digits", "path": "d.csv", "target_encoding": "onehot"},
            "model": {"layers": [784, 10], "output_activation": "softmax"},
            "metrics": [],
        },
        "categorical_crossentropy",
    ),
    "test_path_sin_final_eval": (
        lambda c: with_key(c, "dataset", {"path": "a.csv", "target": "y", "test_path": "b.csv"}),
        "--final-eval",
    ),
    "selected_from_sin_final_eval": (
        lambda c: with_key(c, "selected_from", "results/x/abc"),
        "--final-eval",
    ),
    "tipo_invalido": (lambda c: with_key(c, "training.epochs", "20"), "training.epochs"),
    "bool_no_es_entero": (lambda c: with_key(c, "training.batch_size", True), "batch_size"),
    "salida_incompatible_con_el_target": (
        lambda c: with_key(c, "model.layers", [2, 3, 2]),
        "model.layers",
    ),
    "entrada_incompatible_con_el_dataset": (
        lambda c: with_key(c, "model.layers", [3, 3, 1]),
        "model.layers",
    ),
    "falta_una_obligatoria": (lambda c: {k: v for k, v in c.items() if k != "model"}, "model"),
    "escalon_sin_trainer_perceptron": (
        lambda c: with_key(with_key(c, "model.layers", [2, 1]), "model.output_activation", "step"),
        "perceptron",
    ),
    "early_stopping_val_sin_validacion": (
        lambda c: with_key(c, "training.early_stopping", {"patience": 3}),
        "val_loss",
    ),
    "monitor_que_se_maximiza_con_mode_min": (
        lambda c: with_key(
            with_key(c, "dataset.split", {"kind": "holdout", "ratio": 0.5}),
            "training.early_stopping",
            {"patience": 3, "monitor": "val_accuracy"},
        ),
        "mode",
    ),
    "metrica_desconocida": (lambda c: with_key(c, "metrics", ["auc"]), "metrics"),
    "barrer_una_clave_que_no_entra_al_hash": (
        lambda c: with_key(c, "sweep", {"logging.every": [1, 2]}),
        "logging.every",
    ),
    "seed_y_seeds": (lambda c: with_key(c, "seed", 1), "seed"),
    "valor_invalido_del_optimizador": (
        lambda c: with_key(c, "training.optimizer.lr", -1.0),
        "training.optimizer",
    ),
    "sweep_con_combinacion_invalida": (
        lambda c: with_key(c, "sweep", {"training.optimizer.lr": [0.1, 0.0]}),
        "lr",
    ),
}


@pytest.mark.parametrize("case", list(INVALID))
def test_validacion_rechaza_con_mensaje_que_nombra_la_clave(tmp_path, case):
    mutate, match = INVALID[case]
    path = write_json(tmp_path / "c.json", mutate(xor_config()))
    with pytest.raises(ConfigError, match=match):
        load_config(path)


def test_test_path_se_acepta_con_final_eval():
    config = xor_config(
        dataset={"path": "a.csv", "target": "y", "test_path": "b.csv"},
        model={"layers": ["auto", 1]},
        metrics=[],
        selected_from="results/x/abc",
    )
    resolved = resolve_config(config, final_eval=True)
    assert resolved["final"] == {"mode": "retrain", "epochs": None, "threshold": None}
    with pytest.raises(ConfigError, match="selected_from"):
        resolve_config({**config, "selected_from": None}, final_eval=True)


def test_json_mal_formado_es_config_error(tmp_path):
    path = tmp_path / "c.json"
    path.write_text('{"run_name": "x",}', encoding="utf-8")
    with pytest.raises(ConfigError, match="JSON"):
        load_config(str(path))


# --- expand y config_hash ---


def test_expand_2_claves_x_3_valores_x_2_semillas():
    config = xor_config(
        seeds=[0, 1],
        sweep={
            "training.optimizer.lr": [0.01, 0.1, 1.0],
            "model.hidden_activation": ["tanh", "sigmoid", "relu"],
        },
    )
    runs = expand(resolve_config(config))
    assert len(runs) == 18
    assert all("seeds" not in r and r["fold"] is None for r in runs)

    by_hash: dict[str, list[int]] = {}
    for r in runs:
        by_hash.setdefault(config_hash(r), []).append(r["seed"])
    # Misma config con distinta semilla → mismo hash; cada combinación, uno propio.
    assert len(by_hash) == 9
    assert all(sorted(seeds) == [0, 1] for seeds in by_hash.values())
    combos = {(r["training"]["optimizer"]["lr"], r["model"]["hidden_activation"]) for r in runs}
    assert len(combos) == 9
    assert len({run_id(r) for r in runs}) == 18


def test_expand_kfold_agrega_un_run_por_fold():
    config = xor_config(
        seeds=[0, 1],
        dataset={
            "synthetic": {"name": "line", "n": 20},
            "normalize": "none",
            "split": {"kind": "kfold", "k": 4},
        },
        model={"layers": [1, 1]},
        metrics=[],
    )
    runs = expand(resolve_config(config))
    assert [(r["seed"], r["fold"]) for r in runs] == [(s, f) for s in (0, 1) for f in range(4)]
    assert run_id(runs[5]) == f"{config_hash(runs[5])}_s1_f1"


def test_expand_sweep_de_una_seccion_entera():
    optimizers = [{"kind": "gd", "lr": 0.1}, {"kind": "momentum", "lr": 0.05, "alpha": 0.8}]
    config = resolve_config(xor_config(sweep={"training.optimizer": optimizers}))
    runs = expand(config)
    assert [r["training"]["optimizer"] for r in runs] == optimizers


def test_config_hash_ignora_semilla_nombre_y_logging():
    config = resolve_config(xor_config())
    h = config_hash(config)
    assert len(h) == 8 and int(h, 16) >= 0
    assert config_hash({**config, "seeds": [5], "run_name": "otro"}) == h
    assert config_hash(with_key(config, "logging.every", 1)) == h
    assert config_hash(with_key(config, "training.optimizer.lr", 0.2)) != h


# --- build ---


def test_build_arma_todo_desde_la_config():
    config = xor_config(
        dataset={"synthetic": {"name": "xor"}, "normalize": "none"},
        model={
            "layers": ["auto", 4, 1],
            "hidden_activation": "sigmoid",
            "output_activation": "tanh",
            "beta": 2.0,
            "initializer": "uniform",
            "init_params": {"low": -0.1, "high": 0.1},
        },
        training={
            "optimizer": {"kind": "adam", "lr": 0.02},
            "epochs": 3,
            "early_stopping": {"patience": 4},
            "adaptive_eta": {"a": 0.001, "b": 0.5, "k": 3, "k_prime": 2},
            "augmentation": [{"kind": "gaussian_noise", "sigma": 0.1}],
        },
    )
    run = expand(resolve_config(config))[0]
    network, optimizer, loss, callbacks, metrics, augment = build(
        run, np.random.default_rng(0), n_features=2, n_outputs=1, task="binary", threshold=0.0
    )
    assert network.layer_sizes == [2, 4, 1]
    assert network.layers[0].activation.beta == 2.0 and network.layers[1].activation.beta == 2.0
    assert np.all(np.abs(network.layers[0].W) <= 0.1)
    assert isinstance(optimizer, Adam) and optimizer.lr == 0.02
    assert [type(cb) for cb in callbacks] == [EarlyStopping, AdaptiveEta]
    assert callbacks[1].optimizer is optimizer
    assert list(metrics) == ["accuracy"]
    y = np.array([[1.0], [-1.0]])
    assert metrics["accuracy"](y, np.array([[0.3], [0.2]])) == 0.5
    assert isinstance(augment, Compose) and isinstance(augment.augmentations[0], GaussianNoise)
    assert loss.value(y, y) == 0.0


def test_build_rechaza_datos_que_no_coinciden_con_las_capas():
    run = expand(resolve_config(xor_config()))[0]
    with pytest.raises(ConfigError, match="features"):
        build(run, np.random.default_rng(0), n_features=5)
    with pytest.raises(ConfigError, match="columnas"):
        build(run, np.random.default_rng(0), n_outputs=3)


def test_build_random_shift():
    config = xor_config(
        dataset={"format": "digits", "path": "d.csv", "target_encoding": "onehot"},
        model={"layers": [784, 10]},
        metrics=[],
    )
    config["training"]["augmentation"] = {"kind": "random_shift", "max_px": 1}
    run = expand(resolve_config(config))[0]
    *_, augment = build(run, np.random.default_rng(0))
    assert isinstance(augment, RandomShift) and augment.image_shape == (28, 28)


# --- Runner: helpers ---


def run_dir_of(root, config: dict):
    """Carpeta de la primera corrida de config dentro de root."""
    return root / config["run_name"] / run_id(expand(resolve_config(config))[0])


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_binary_csv(path, n=200, seed=0):
    """CSV tabular: dos features y una etiqueta 0/1 dada por una recta, con algo de ruido."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.normal(size=n) > 0).astype(int)
    lines = ["x1,x2,label"] + [f"{a:.6f},{b:.6f},{c}" for (a, b), c in zip(X, y, strict=True)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def write_digits_csv(path, labels, seed=0):
    images = np.random.default_rng(seed).uniform(0, 1, size=(len(labels), 784)).round(3)
    lines = ["label,image"]
    for label, img in zip(labels, images, strict=True):
        lines.append(f'{label},"{json.dumps(img.tolist())}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def line_config(optimizer: dict, **sections) -> dict:
    """y = tanh(x) con holdout, mini-batch, early stopping y η adaptativo."""
    config = {
        "run_name": "linea",
        "seeds": [3],
        "dataset": {
            "synthetic": {"name": "line", "function": "tanh", "n": 40, "low": -2.0, "high": 2.0},
            "normalize": "none",
            "split": {"kind": "holdout", "ratio": 0.75, "stratified": False},
        },
        "model": {"layers": [1, 4, 1], "output_activation": "tanh"},
        "training": {
            "optimizer": optimizer,
            "batch_size": 8,
            "epochs": 10,
            "early_stopping": {"patience": 50},
            "adaptive_eta": {"a": 0.01, "b": 0.5, "k": 2, "k_prime": 2},
        },
        "metrics": ["mse"],
        "logging": {"every": 5, "checkpoint_every": 5},
    }
    config.update(sections)
    return config


class CrashAt:
    """Callback que simula un corte (el proceso muere) al terminar una época."""

    def __init__(self, epoch):
        self.epoch = epoch

    def on_train_begin(self, network):
        pass

    def on_epoch_end(self, epoch, logs, network):
        if epoch == self.epoch:
            raise KeyboardInterrupt("corte simulado")
        return False

    def on_train_end(self, network):
        pass


# --- Runner: corrida end-to-end ---


def test_corrida_end_to_end_de_xor_genera_todos_los_archivos(tmp_path):
    config = xor_config(logging={"every": 10, "save_predictions": True})
    config["training"]["epochs"] = 30
    path = write_json(tmp_path / "xor.json", config)
    lines: list[str] = []
    results = run_experiment(path, results_dir=tmp_path / "results", write=lines.append)

    run_dir = run_dir_of(tmp_path / "results", config)
    assert [r["status"] for r in results] == ["ok"]
    assert results[0]["run_dir"] == str(run_dir)
    for name in ("config.json", "history.csv", "metrics.json", "predictions.npz", "model.npz"):
        assert (run_dir / name).exists(), name
    assert not (run_dir / "checkpoint.npz").exists()  # se borra al terminar
    assert (tmp_path / "results" / "xor" / "log.txt").exists()

    assert read_json(run_dir / "config.json") == expand(resolve_config(config))[0]
    header, *rows = (run_dir / "history.csv").read_text().splitlines()
    assert len(rows) == 30 and "train_accuracy" in header.split(",")

    metrics = read_json(run_dir / "metrics.json")
    assert metrics["status"] == "ok" and metrics["run_id"] == run_dir.name
    assert metrics["epochs_trained"] == 30 and metrics["best_epoch"] == 30
    assert metrics["n_params"] == 2 * 3 + 3 + 3 * 1 + 1
    assert metrics["task"] == "binary" and metrics["threshold"] == 0.0
    assert isinstance(metrics["git_commit"], str) and len(metrics["data"]["sha256"]) == 64
    assert metrics["time_total_s"] > 0 and metrics["s_per_epoch"] > 0
    assert metrics["val"] is None
    train = metrics["train"]
    assert 0.0 <= train["accuracy"] <= 1.0 and np.array(train["confusion_matrix"]).shape == (2, 2)
    assert train["loss"] == pytest.approx(float(rows[-1].split(",")[1]))

    with np.load(run_dir / "predictions.npz") as preds:
        assert str(preds["subset"]) == "train"  # sin validación: las de train
        np.testing.assert_array_equal(preds["idx"], np.arange(4))
        y_score = preds["y_score"]
    network, optimizer, extra = load_checkpoint(run_dir / "model.npz")
    X = np.array([[-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [1.0, 1.0]])
    np.testing.assert_allclose(network.predict(X), y_score)
    assert optimizer.lr == 0.1 and extra["preprocessing"]["task"] == "binary"
    assert extra["config"] == read_json(run_dir / "config.json")

    assert any(line.startswith("[1/1]") for line in lines)
    assert any("epoch 10:" in line for line in lines)


def test_con_validacion_guarda_predicciones_de_validacion(tmp_path):
    config = line_config({"kind": "gd", "lr": 0.1}, logging={"save_predictions": True})
    run_experiment(write_json(tmp_path / "c.json", config), results_dir=tmp_path, write=print)
    run_dir = run_dir_of(tmp_path, config)
    metrics = read_json(run_dir / "metrics.json")
    assert metrics["data"]["n_train"] == 30 and metrics["data"]["n_val"] == 10
    assert set(metrics["val"]) >= {"loss", "mse", "rmse", "mae"}
    with np.load(run_dir / "predictions.npz") as preds:
        assert str(preds["subset"]) == "val" and preds["y_score"].shape == (10, 1)


# --- Resume ---


@pytest.mark.parametrize(
    "optimizer", [{"kind": "gd", "lr": 0.1}, {"kind": "adam", "lr": 0.01}], ids=["gd", "adam"]
)
def test_resume_identico_a_no_cortar(tmp_path, optimizer):
    config = line_config(optimizer)
    path = write_json(tmp_path / "c.json", config)

    # Corrida de 10 épocas sin cortar.
    run_experiment(path, results_dir=tmp_path / "a", write=print)
    straight = run_dir_of(tmp_path / "a", config)

    # La misma corrida, cortada en la época 6: queda el checkpoint de la 5.
    run = expand(resolve_config(config))[0]
    cut = tmp_path / "b" / run_id(run)
    with pytest.raises(KeyboardInterrupt):
        train_run(run, cut, write=print, callbacks=[CrashAt(6)])
    assert not (cut / "metrics.json").exists()
    _, _, extra = load_checkpoint(cut / "checkpoint.npz")
    assert extra["epoch"] == 5 and len(extra["history"]) == 5

    lines: list[str] = []
    result = resume_run(cut, write=lines.append)
    assert result["status"] == "ok"
    assert any("epoch 10:" in line for line in lines)  # épocas absolutas

    net_a, opt_a, _ = load_checkpoint(straight / "model.npz")
    net_b, opt_b, _ = load_checkpoint(cut / "model.npz")
    for a, b in zip(net_a.params, net_b.params, strict=True):
        np.testing.assert_allclose(b, a, rtol=0, atol=1e-12)
    state_a, state_b = opt_a.state_dict(), opt_b.state_dict()
    assert state_b.pop("kind") == state_a.pop("kind")
    assert state_b["t"] == state_a["t"] > 0
    for key in state_a:
        np.testing.assert_allclose(state_b[key], state_a[key], rtol=0, atol=1e-12)

    hist_a = np.genfromtxt(straight / "history.csv", delimiter=",", names=True)
    hist_b = np.genfromtxt(cut / "history.csv", delimiter=",", names=True)
    for column in hist_a.dtype.names:
        if column != "elapsed_s":
            np.testing.assert_allclose(hist_b[column], hist_a[column], rtol=0, atol=1e-12)
    assert np.all(np.diff(hist_b["elapsed_s"]) >= 0)

    metrics_a, metrics_b = read_json(straight / "metrics.json"), read_json(cut / "metrics.json")
    assert metrics_b["resumed_from_epoch"] == 5 and metrics_a["resumed_from_epoch"] is None
    assert metrics_b["best_epoch"] == metrics_a["best_epoch"]
    assert metrics_b["val"]["loss"] == pytest.approx(metrics_a["val"]["loss"], abs=1e-12)


def test_resume_de_una_corrida_terminada_no_hace_nada(tmp_path):
    config = xor_config()
    run_experiment(write_json(tmp_path / "c.json", config), results_dir=tmp_path, write=print)
    run_dir = run_dir_of(tmp_path, config)
    before = (run_dir / "metrics.json").stat().st_mtime_ns
    assert resume_run(run_dir, write=print)["status"] == "skipped"
    assert (run_dir / "metrics.json").stat().st_mtime_ns == before


def test_resume_sin_checkpoint_es_error(tmp_path):
    run = expand(resolve_config(xor_config()))[0]
    run_dir = tmp_path / run_id(run)
    run_dir.mkdir()
    write_json(run_dir / "config.json", run)
    with pytest.raises(RunnerError, match="checkpoint"):
        resume_run(run_dir, write=print)


# --- Barridos ---


def test_relanzar_un_barrido_no_reentrena(tmp_path):
    config = xor_config(seeds=[0, 1], sweep={"training.optimizer.lr": [0.05, 0.1]})
    path = write_json(tmp_path / "c.json", config)
    first = run_experiment(path, results_dir=tmp_path, write=print)
    assert [r["status"] for r in first] == ["ok"] * 4

    def stamp(result):
        return (tmp_path / "xor" / result["run_id"] / "metrics.json").stat().st_mtime_ns

    stamps = {r["run_id"]: stamp(r) for r in first}
    second = run_experiment(path, results_dir=tmp_path, write=print)
    assert [r["status"] for r in second] == ["skipped"] * 4
    assert all(stamp(r) == stamps[r["run_id"]] for r in second)

    forced = run_experiment(path, results_dir=tmp_path, force=True, only=["seed=1"], write=print)
    assert [r["status"] for r in forced] == ["ok"] * 2
    log = (tmp_path / "xor" / "log.txt").read_text(encoding="utf-8")
    assert "[4/4]" in log and "salteada" in log


def test_only_filtra_corridas(tmp_path):
    config = xor_config(seeds=[0, 1], sweep={"training.optimizer.lr": [0.05, 0.1]})
    path = write_json(tmp_path / "c.json", config)
    results = run_experiment(
        path, results_dir=tmp_path, only=["training.optimizer.lr=0.1", "seed=1"], write=print
    )
    assert len(results) == 1
    config_run = read_json(tmp_path / "xor" / results[0]["run_id"] / "config.json")
    assert config_run["seed"] == 1 and config_run["training"]["optimizer"]["lr"] == 0.1
    with pytest.raises(ConfigError, match="ninguna"):
        run_experiment(path, results_dir=tmp_path, only=["seed=9"], write=print)
    with pytest.raises(ConfigError, match="clave=valor"):
        run_experiment(path, results_dir=tmp_path, only=["seed"], write=print)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # overflow esperado de NumPy
def test_una_corrida_que_diverge_se_marca_y_el_barrido_sigue(tmp_path):
    config = {
        "run_name": "diverge",
        "seeds": [0],
        "dataset": {"synthetic": {"name": "line"}, "normalize": "none", "split": {"kind": "none"}},
        "model": {"layers": [1, 1]},
        "training": {"optimizer": {"kind": "gd", "lr": 1e6}, "epochs": 50},
        "sweep": {"training.optimizer.lr": [1e6, 0.1]},
    }
    path = write_json(tmp_path / "c.json", config)
    results = run_experiment(path, results_dir=tmp_path, write=print)
    assert [r["status"] for r in results] == ["diverged", "ok"]
    diverged = read_json(tmp_path / "diverge" / results[0]["run_id"] / "metrics.json")
    assert diverged["status"] == "diverged" and diverged["epochs_trained"] < 50


def test_un_error_en_una_corrida_no_frena_el_barrido(tmp_path):
    config = xor_config(
        dataset={"path": str(tmp_path / "no_existe.csv"), "target": "y"},
        model={"layers": ["auto", 1]},
        metrics=[],
        seeds=[0, 1],
    )
    path = write_json(tmp_path / "c.json", config)
    results = run_experiment(path, results_dir=tmp_path, write=print)
    assert [r["status"] for r in results] == ["error", "error"]
    assert "no_existe.csv" in results[0]["error"]
    assert (tmp_path / "xor" / results[0]["run_id"] / "error.txt").exists()


def test_workers_da_lo_mismo_que_secuencial(tmp_path):
    config = xor_config(seeds=[0, 1], sweep={"training.optimizer.lr": [0.05, 0.1]})
    path = write_json(tmp_path / "c.json", config)
    parallel = run_experiment(path, results_dir=tmp_path / "par", workers=2, write=print)
    sequential = run_experiment(path, results_dir=tmp_path / "seq", write=print)
    assert [r["status"] for r in parallel] == ["ok"] * 4
    assert [r["run_id"] for r in parallel] == [r["run_id"] for r in sequential]
    for r in parallel:
        par = read_json(tmp_path / "par" / "xor" / r["run_id"] / "metrics.json")
        seq = read_json(tmp_path / "seq" / "xor" / r["run_id"] / "metrics.json")
        assert par["train"]["loss"] == seq["train"]["loss"]


def test_trainer_perceptron_aprende_and(tmp_path):
    config = {
        "run_name": "and",
        "seeds": [0, 1, 2],
        "trainer": "perceptron",
        "dataset": {"synthetic": {"name": "and"}, "normalize": "none", "split": {"kind": "none"}},
        "model": {"layers": [2, 1], "output_activation": "step", "initializer": "uniform"},
        "training": {"optimizer": {"lr": 0.1}, "epochs": 20},
        "metrics": ["accuracy"],
    }
    results = run_experiment(write_json(tmp_path / "c.json", config), results_dir=tmp_path)
    for r in results:
        metrics = read_json(tmp_path / "and" / r["run_id"] / "metrics.json")
        assert metrics["train"]["accuracy"] == 1.0 and metrics["train"]["loss"] == 0.0


def test_formato_digits_end_to_end(tmp_path):
    labels = [c for c in range(10) for _ in range(6)]
    config = {
        "run_name": "digitos",
        "seeds": [0],
        "dataset": {
            "format": "digits",
            "path": write_digits_csv(tmp_path / "digits.csv", labels),
            "normalize": "none",
            "target_encoding": "onehot",
            "split": {"kind": "holdout", "ratio": 0.5},
        },
        "model": {"layers": [784, 8, 10], "output_activation": "softmax"},
        "training": {
            "loss": "categorical_crossentropy",
            "optimizer": {"kind": "adam", "lr": 0.01},
            "batch_size": 16,
            "epochs": 3,
            "augmentation": {"kind": "random_shift", "max_px": 1},
        },
        "metrics": ["accuracy", "macro_f1"],
    }
    results = run_experiment(write_json(tmp_path / "c.json", config), results_dir=tmp_path)
    metrics = read_json(tmp_path / "digitos" / results[0]["run_id"] / "metrics.json")
    assert metrics["task"] == "multiclass" and metrics["n_classes"] == 10
    assert metrics["data"]["n_train"] == 30 and metrics["data"]["n_val"] == 30
    assert np.array(metrics["val"]["confusion_matrix"]).shape == (10, 10)
    assert np.array(metrics["val"]["per_class"]["recall"]).shape == (10,)


# --- Smoke ---


def test_smoke_limita_epocas_y_muestras(tmp_path):
    config = {
        "run_name": "tabular",
        "dataset": {
            "path": write_binary_csv(tmp_path / "datos.csv", n=1000),
            "target": "label",
            "split": {"kind": "kfold", "k": 3},
        },
        "model": {"layers": ["auto", 4, 1], "output_activation": "sigmoid"},
        "training": {"optimizer": {"kind": "adam", "lr": 0.01}, "epochs": 100},
        "metrics": ["accuracy", "f1"],
        "sweep": {"training.batch_size": [16, 64]},
    }
    lines: list[str] = []
    path = write_json(tmp_path / "c.json", config)
    results = run_experiment(path, results_dir=tmp_path, smoke=True, write=lines.append)
    assert len(results) == 1 and results[0]["status"] == "ok"
    run_dir = tmp_path / "_smoke" / "tabular" / results[0]["run_id"]
    metrics = read_json(run_dir / "metrics.json")
    assert metrics["smoke"] is True and metrics["epochs_trained"] == 5
    assert metrics["data"]["n_train"] + metrics["data"]["n_val"] == 500
    assert metrics["data"]["n_dev"] == 1000
    assert not (tmp_path / "tabular").exists()
    # Estimación: 2 valores de batch × 3 semillas × 3 folds = 18 corridas.
    assert any("18 corridas" in line for line in lines)


VALIDATION_CONFIGS = sorted(
    (Path(__file__).parents[1] / "experiments/configs/validacion").glob("*.json")
)


def test_estan_las_cinco_configs_de_validacion():
    names = [p.stem for p in VALIDATION_CONFIGS]
    assert names == ["and_step", "linear_identity", "nonlinear_tanh", "xor_221", "xor_2321"]


def test_configs_de_validacion_corren_con_smoke_en_menos_de_60_s(tmp_path):
    start = time.perf_counter()
    for path in VALIDATION_CONFIGS:
        results = run_experiment(path, results_dir=tmp_path, smoke=True, write=lambda line: None)
        assert [r["status"] for r in results] == ["ok"], path.name
        metrics = read_json(Path(results[0]["run_dir"]) / "metrics.json")
        assert metrics["epochs_trained"] <= 5
    assert time.perf_counter() - start < 60


# --- holdout_test y --final-eval ---


def dev_config(csv: str, indices: str, **sections) -> dict:
    config = {
        "run_name": "fraude_cv",
        "seeds": [0, 1],
        "dataset": {
            "path": csv,
            "target": "label",
            "normalize": "zscore",
            "split": {"kind": "holdout", "ratio": 0.8},
            "holdout_test": {"ratio": 0.25, "seed": 7, "indices_path": indices},
        },
        "model": {"layers": ["auto", 1], "output_activation": "sigmoid"},
        "training": {
            "optimizer": {"kind": "adam", "lr": 0.05},
            "batch_size": 16,
            "epochs": 30,
            "early_stopping": {"patience": 5},
        },
        "metrics": ["accuracy"],
        "logging": {"save_predictions": True},
    }
    config.update(sections)
    return config


def test_holdout_test_excluye_los_indices_de_test_y_final_eval_los_usa(tmp_path):
    csv = write_binary_csv(tmp_path / "datos.csv", n=200)
    indices = str(tmp_path / "split" / "test_idx.npy")
    results = run_experiment(write_json(tmp_path / "dev.json", dev_config(csv, indices)),
                             results_dir=tmp_path)  # fmt: skip
    test_idx = np.load(indices)
    assert len(test_idx) == 50 and len(np.unique(test_idx)) == 50
    best_epochs = []
    for r in results:
        run_dir = tmp_path / "fraude_cv" / r["run_id"]
        metrics = read_json(run_dir / "metrics.json")
        assert metrics["data"]["n_train"] + metrics["data"]["n_val"] == 150
        assert metrics["data"]["n_test_excluded"] == 50
        with np.load(run_dir / "predictions.npz") as preds:
            assert np.intersect1d(preds["idx"], test_idx).size == 0
        best_epochs.append(metrics["best_epoch"])

    # --final-eval: reentrena con todo desarrollo durante la mediana de best_epoch.
    hash8 = results[0]["run_id"].split("_")[0]
    final = dev_config(
        csv, indices, run_name="fraude_final", selected_from=str(tmp_path / "fraude_cv" / hash8)
    )
    final_path = write_json(tmp_path / "final.json", final)
    with pytest.raises(ConfigError):  # sin --final-eval no se acepta selected_from
        run_experiment(final_path, results_dir=tmp_path)
    summary = final_eval(final_path, results_dir=tmp_path)

    report = read_json(tmp_path / "fraude_final" / "final_eval.json")
    assert report == json.loads(json.dumps(summary))
    assert report["mode"] == "retrain" and report["epochs"] == int(np.round(np.median(best_epochs)))
    assert report["test"]["source"] == "holdout_test" and report["test"]["n"] == 50
    assert sorted(report["selected_runs"]) == sorted(r["run_id"] for r in results)
    assert [m["label"] for m in report["models"]] == ["s0", "s1"]
    assert np.array(report["models"][0]["test"]["confusion_matrix"]).sum() == 50
    assert 0 <= report["mean"]["accuracy"] <= 1 and "accuracy" in report["std"]
    for seed in (0, 1):
        final_run = read_json(tmp_path / "fraude_final" / f"final_s{seed}" / "metrics.json")
        assert final_run["data"]["n_train"] == 150 and final_run["val"] is None
        assert final_run["epochs_trained"] == report["epochs"]
    assert "lectura del test" in (tmp_path / "fraude_final" / "log.txt").read_text("utf-8")

    # El test se evalúa una sola vez.
    with pytest.raises(RunnerError, match="una sola vez"):
        final_eval(final_path, results_dir=tmp_path)


def test_final_eval_con_test_path_reusando_modelos(tmp_path):
    config = dev_config(write_binary_csv(tmp_path / "dev.csv", n=120), "", seeds=[0])
    del config["dataset"]["holdout_test"]
    results = run_experiment(write_json(tmp_path / "dev.json", config), results_dir=tmp_path)
    selected = tmp_path / "fraude_cv" / results[0]["run_id"]

    final = copy.deepcopy(config)
    final["run_name"] = "fraude_final"
    final["dataset"]["test_path"] = write_binary_csv(tmp_path / "test.csv", n=40, seed=1)
    final["selected_from"] = str(selected)
    final["final"] = {"mode": "reuse", "threshold": 0.3}
    report = final_eval(write_json(tmp_path / "final.json", final), results_dir=tmp_path)
    assert report["mode"] == "reuse" and report["epochs"] is None
    assert report["test"]["source"] == "test_path" and report["test"]["n"] == 40
    assert report["threshold"] == 0.3
    assert [m["label"] for m in report["models"]] == [selected.name]
    out = tmp_path / "fraude_final" / f"final_{selected.name}" / "test_predictions.npz"
    with np.load(out) as preds:
        labels = (preds["y_score"][:, 0] >= 0.3).astype(int)
        np.testing.assert_array_equal(preds["labels_pred"], labels)


# --- CLI ---


def test_cli(tmp_path, capsys):
    path = write_json(tmp_path / "c.json", xor_config())
    assert main([path, "--results-dir", str(tmp_path)]) == 0
    assert main([path, "--results-dir", str(tmp_path)]) == 0  # todo salteado
    bad = write_json(tmp_path / "bad.json", with_key(xor_config(), "training.epochs", 0))
    assert main([bad, "--results-dir", str(tmp_path)]) == 2
    assert "training.epochs" in capsys.readouterr().err
    assert main(["--resume", str(tmp_path / "no_existe")]) == 2
    with pytest.raises(SystemExit):
        main([path, "--smoke", "--final-eval"])
