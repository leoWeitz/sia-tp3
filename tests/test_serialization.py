"""Guardado y carga de redes, optimizadores y estado de entrenamiento (F07).

Cubre core/serialization.py (save_checkpoint, load_checkpoint, Checkpoint) y
el state_dict de los callbacks con estado (EarlyStopping, AdaptiveEta), que
el runner usa para reanudar un entrenamiento cortado.
"""

import json

import numpy as np
import pytest

from core.activations import Sigmoid, Tanh
from core.callbacks import AdaptiveEta, EarlyStopping
from core.initializers import Uniform
from core.losses import MSE
from core.network import Network
from core.optimizers import GD, Adam, Momentum
from core.serialization import Checkpoint, load_checkpoint, save_checkpoint


def regression_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(40, 3))
    y = np.tanh(X @ np.array([[1.0], [-2.0], [0.5]]))
    return X, y


def mlp(seed: int = 0) -> Network:
    return Network(
        [3, 5, 4, 1],
        hidden_activation=Tanh(beta=2.0),
        output_activation=Sigmoid(beta=0.5),
        initializer=Uniform(-0.3, 0.3),
        rng=np.random.default_rng(seed),
    )


# --- Estado de los callbacks ---


def feed(callback, values, monitor="val_loss", start=1, network=None):
    """Pasa una secuencia de valores de monitor al callback; devuelve lo que respondió."""
    return [
        callback.on_epoch_end(epoch, {monitor: v}, network)
        for epoch, v in enumerate(values, start=start)
    ]


def test_early_stopping_state_dict_continua_igual_que_sin_cortar():
    values = [1.0, 0.5, 0.6, 0.4, 0.45, 0.47, 0.5]
    net = Network([1, 1], rng=np.random.default_rng(0))

    straight = EarlyStopping(patience=3)
    straight.on_train_begin(net)
    expected = feed(straight, values, network=net)

    first = EarlyStopping(patience=3)
    first.on_train_begin(net)
    feed(first, values[:4], network=net)
    state = first.state_dict()

    resumed = EarlyStopping(patience=3)
    resumed.on_train_begin(net)  # reinicia: por eso load_state_dict va después
    resumed.load_state_dict(state)
    got = feed(resumed, values[4:], start=5, network=net)

    assert got == expected[4:]
    assert (resumed.best, resumed.best_epoch, resumed.stopped_epoch) == (
        straight.best,
        straight.best_epoch,
        straight.stopped_epoch,
    )


def test_early_stopping_state_dict_copia_los_mejores_pesos():
    net = Network([1, 1], rng=np.random.default_rng(0))
    stopper = EarlyStopping(patience=2)
    stopper.on_train_begin(net)
    stopper.on_epoch_end(1, {"val_loss": 1.0}, net)
    state = stopper.state_dict()
    for p in net.params:
        p[...] = 99.0

    resumed = EarlyStopping(patience=2)
    resumed.on_train_begin(net)
    resumed.load_state_dict(state)
    state["best_params"][0][...] = -1.0  # el callback guarda su propia copia
    resumed.on_train_end(net)
    assert not np.any(net.params[0] == 99.0)
    assert not np.any(net.params[0] == -1.0)


def test_early_stopping_state_dict_es_serializable_salvo_los_pesos():
    stopper = EarlyStopping(patience=2)
    state = stopper.state_dict()  # antes de entrenar también tiene estado
    assert state["best"] == np.inf and state["best_epoch"] is None
    json.dumps({k: v for k, v in state.items() if k != "best_params"})


def test_adaptive_eta_state_dict_continua_igual_que_sin_cortar():
    losses = [1.0, 0.9, 0.8, 0.85, 0.9, 0.95, 0.7, 0.6]

    def run(values, eta, start=1):
        for epoch, v in enumerate(values, start=start):
            eta.on_epoch_end(epoch, {"train_loss": v}, None)

    straight = AdaptiveEta(GD(lr=1.0), a=0.1, b=0.5, k=2, k_prime=3)
    straight.on_train_begin(None)
    run(losses, straight)

    opt = GD(lr=1.0)
    first = AdaptiveEta(opt, a=0.1, b=0.5, k=2, k_prime=3)
    first.on_train_begin(None)
    run(losses[:4], first)
    state = first.state_dict()
    json.dumps(state)

    resumed = AdaptiveEta(opt, a=0.1, b=0.5, k=2, k_prime=3)
    resumed.on_train_begin(None)
    resumed.load_state_dict(state)
    run(losses[4:], resumed, start=5)
    assert opt.lr == pytest.approx(straight.optimizer.lr)


# --- save_checkpoint / load_checkpoint ---


def test_save_load_mismas_predicciones_y_arquitectura(tmp_path):
    X, y = regression_data()
    net = mlp()
    net.fit(X, y, GD(lr=0.1), MSE(), epochs=5)
    path = tmp_path / "model.npz"
    save_checkpoint(path, net)

    loaded, optimizer, extra = load_checkpoint(path)
    assert optimizer is None and extra == {}
    assert loaded.layer_sizes == net.layer_sizes
    np.testing.assert_array_equal(loaded.predict(X), net.predict(X))
    hidden, output = loaded.layers[0].activation, loaded.layers[-1].activation
    assert isinstance(hidden, Tanh) and hidden.beta == 2.0
    assert isinstance(output, Sigmoid) and output.beta == 0.5
    assert isinstance(loaded.initializer, Uniform)
    assert (loaded.initializer.low, loaded.initializer.high) == (-0.3, 0.3)


def test_el_optimizador_conserva_su_estado(tmp_path):
    X, y = regression_data()
    net, opt = mlp(), Adam(lr=0.01)
    net.fit(X, y, opt, MSE(), epochs=3, batch_size=8)
    save_checkpoint(tmp_path / "ckpt.npz", net, opt)

    _, loaded_opt, _ = load_checkpoint(tmp_path / "ckpt.npz")
    assert isinstance(loaded_opt, Adam)
    expected = opt.state_dict()
    got = loaded_opt.state_dict()
    assert set(got) == set(expected)
    for key, value in expected.items():
        np.testing.assert_array_equal(got[key], value)


@pytest.mark.parametrize(
    "make_opt", [lambda: GD(lr=0.05), lambda: Adam(lr=0.01)], ids=["gd", "adam"]
)
def test_cargar_y_seguir_entrenando_es_identico_a_no_cortar(tmp_path, make_opt):
    X, y = regression_data()
    straight, opt = mlp(), make_opt()
    straight.fit(X, y, opt, MSE(), epochs=4, batch_size=8)
    straight.fit(X, y, opt, MSE(), epochs=4, batch_size=8)

    cut, opt_cut = mlp(), make_opt()
    cut.fit(X, y, opt_cut, MSE(), epochs=4, batch_size=8)
    save_checkpoint(tmp_path / "ckpt.npz", cut, opt_cut)
    resumed, opt_resumed, _ = load_checkpoint(tmp_path / "ckpt.npz")
    # Mismo rng (orden de los lotes) y mismo estado del optimizador.
    resumed.fit(X, y, opt_resumed, MSE(), epochs=4, batch_size=8)

    for a, b in zip(resumed.params, straight.params, strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def test_el_rng_de_la_red_se_restaura(tmp_path):
    net = mlp()
    net.rng.permutation(10)  # avanza el generador
    save_checkpoint(tmp_path / "ckpt.npz", net)
    loaded, _, _ = load_checkpoint(tmp_path / "ckpt.npz")
    np.testing.assert_array_equal(loaded.rng.permutation(100), net.rng.permutation(100))


def test_la_red_cargada_es_entrenable(tmp_path):
    X, y = regression_data()
    save_checkpoint(tmp_path / "ckpt.npz", mlp())
    loaded, _, _ = load_checkpoint(tmp_path / "ckpt.npz")
    before = [p.copy() for p in loaded.params]
    history = loaded.fit(X, y, Momentum(lr=0.1), MSE(), epochs=5)
    assert history.column("train_loss")[-1] < history.column("train_loss")[0]
    assert not np.array_equal(loaded.params[0], before[0])


def test_extra_ida_y_vuelta_con_arrays_anidados(tmp_path):
    big = np.random.default_rng(3).bit_generator.state  # enteros de 128 bits
    extra = {
        "epoch": 7,
        "best": np.inf,
        "nan": np.nan,
        "name": "xor",
        "nested": {"arrays": [np.arange(3.0), np.eye(2)], "none": None, "flag": True},
        "rng": big,
        "np_scalar": np.int64(5),
    }
    save_checkpoint(tmp_path / "ckpt.npz", mlp(), extra=extra)
    _, _, got = load_checkpoint(tmp_path / "ckpt.npz")

    assert got["epoch"] == 7 and got["best"] == np.inf and np.isnan(got["nan"])
    assert got["name"] == "xor" and got["np_scalar"] == 5
    np.testing.assert_array_equal(got["nested"]["arrays"][0], np.arange(3.0))
    np.testing.assert_array_equal(got["nested"]["arrays"][1], np.eye(2))
    assert got["nested"]["none"] is None and got["nested"]["flag"] is True
    assert got["rng"] == big


def test_perceptron_escalon_ida_y_vuelta(tmp_path):
    net = Network([2, 1], output_activation="step", rng=np.random.default_rng(0))
    save_checkpoint(tmp_path / "ckpt.npz", net)
    loaded, _, _ = load_checkpoint(tmp_path / "ckpt.npz")
    X = np.array([[-1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_array_equal(loaded.predict(X), net.predict(X))


def test_se_guarda_sin_pickle_y_sin_archivos_temporales(tmp_path):
    save_checkpoint(tmp_path / "ckpt.npz", mlp(), Adam(lr=0.01), extra={"a": [1, 2]})
    with np.load(tmp_path / "ckpt.npz", allow_pickle=False) as data:
        assert "W_0" in data and "b_2" in data
    assert [p.name for p in tmp_path.iterdir()] == ["ckpt.npz"]


def test_activacion_fuera_del_registro_es_error(tmp_path):
    class Custom:
        def forward(self, z):
            return z

        def backward(self, z):
            return np.ones_like(z)

    net = Network([2, 1], output_activation=Custom(), rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="Custom"):
        save_checkpoint(tmp_path / "ckpt.npz", net)


# --- Callback Checkpoint ---


def test_checkpoint_guarda_cada_n_epocas_con_extra(tmp_path):
    X, y = regression_data()
    path = tmp_path / "checkpoint.npz"
    saved = []

    def extra(epoch, logs):
        saved.append(epoch)
        return {"epoch": epoch, "train_loss": logs["train_loss"]}

    net, opt = mlp(), Adam(lr=0.01)
    checkpoint = Checkpoint(path, every=3, optimizer=opt, extra=extra)
    history = net.fit(X, y, opt, MSE(), epochs=7, callbacks=[checkpoint])
    assert saved == [3, 6]
    loaded, loaded_opt, got = load_checkpoint(path)
    assert got["epoch"] == 6
    assert got["train_loss"] == pytest.approx(history.column("train_loss")[5])
    assert loaded_opt.t == 6  # lote completo: un paso por época


def test_checkpoint_every_invalido():
    with pytest.raises(ValueError):
        Checkpoint("x.npz", every=0)
