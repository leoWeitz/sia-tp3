import csv

import numpy as np
import pytest

from core.callbacks import AdaptiveEta, Callback, EarlyStopping, PrintProgress
from core.history import COLUMNS, History
from core.losses import MSE
from core.network import Network
from core.optimizers import GD, Adam


def linear_data(n: int = 50, seed: int = 0):
    """y = x con n muestras en [-1, 1], forma (n, 1)."""
    X = np.random.default_rng(seed).uniform(-1, 1, size=(n, 1))
    return X, X.copy()


def linear_net(seed: int = 0) -> Network:
    return Network([1, 1], output_activation="identity", rng=np.random.default_rng(seed))


def mean_abs_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


# --- Criterio de aceptación de la Etapa 4 ---


def test_aceptacion_y_igual_x_mse_baja_monotonamente_y_termina_bajo_1e4():
    X, y = linear_data(50)
    history = linear_net().fit(X, y, GD(lr=0.1), MSE(), epochs=300)

    loss = history.column("train_loss")
    assert np.all(np.diff(loss) <= 0), "el MSE subió en alguna época"
    assert loss[-1] < 1e-4


@pytest.mark.parametrize("batch_size", [1, 8], ids=["estocastico", "mini_batch"])
def test_estocastico_y_mini_batch_tambien_convergen(batch_size):
    X, y = linear_data(50)
    history = linear_net().fit(X, y, GD(lr=0.05), MSE(), epochs=100, batch_size=batch_size)
    assert history.column("train_loss")[-1] < 1e-4


# --- Semántica de fit ---


def test_fit_lote_completo_equivale_al_paso_manual():
    X, y = linear_data(20)
    by_fit, by_hand = linear_net(), linear_net()
    by_fit.fit(X, y, GD(lr=0.1), MSE(), epochs=5)

    loss, opt = MSE(), GD(lr=0.1)
    for _ in range(5):
        by_hand.backward(y, by_hand.predict(X), loss)
        opt.step(by_hand.params, by_hand.grads)

    for a, b in zip(by_fit.params, by_hand.params, strict=True):
        np.testing.assert_array_equal(a, b)


def test_batch_size_mayor_o_igual_a_n_es_lote_completo():
    X, y = linear_data(20)
    h_none = linear_net().fit(X, y, GD(lr=0.1), MSE(), epochs=5)
    h_n = linear_net().fit(X, y, GD(lr=0.1), MSE(), epochs=5, batch_size=20)
    h_big = linear_net().fit(X, y, GD(lr=0.1), MSE(), epochs=5, batch_size=1000)
    np.testing.assert_array_equal(h_none.column("train_loss"), h_n.column("train_loss"))
    np.testing.assert_array_equal(h_none.column("train_loss"), h_big.column("train_loss"))


def test_cantidad_de_pasos_por_epoca():
    class CountingGD(GD):
        steps = 0

        def step(self, params, grads):
            CountingGD.steps += 1
            super().step(params, grads)

    X, y = linear_data(10)
    linear_net().fit(X, y, CountingGD(lr=0.1), MSE(), epochs=3, batch_size=4)
    # 10 muestras en lotes de 4: 4 + 4 + 2 = 3 pasos por época.
    assert CountingGD.steps == 9


def test_misma_semilla_mismo_entrenamiento():
    X, y = linear_data(30)
    h1 = linear_net(7).fit(X, y, GD(lr=0.05), MSE(), epochs=10, batch_size=4)
    h2 = linear_net(7).fit(X, y, GD(lr=0.05), MSE(), epochs=10, batch_size=4)
    np.testing.assert_array_equal(h1.column("train_loss"), h2.column("train_loss"))


def test_rng_explicito_cambia_el_orden_de_los_lotes():
    X, y = linear_data(30)
    h1 = linear_net().fit(X, y, GD(lr=0.05), MSE(), epochs=3, batch_size=4,
                          rng=np.random.default_rng(1))  # fmt: skip
    h2 = linear_net().fit(X, y, GD(lr=0.05), MSE(), epochs=3, batch_size=4,
                          rng=np.random.default_rng(2))  # fmt: skip
    assert not np.array_equal(h1.column("train_loss"), h2.column("train_loss"))


def test_logs_con_validacion_y_metrica():
    X, y = linear_data(40)
    X_val, y_val = linear_data(10, seed=1)
    history = linear_net().fit(X, y, GD(lr=0.1), MSE(), epochs=3,
                               validation=(X_val, y_val), metric=mean_abs_error)  # fmt: skip

    assert len(history) == 3
    record = history.records[-1]
    assert list(record) == list(COLUMNS)
    assert record["epoch"] == 3

    assert np.all(np.isfinite(history.column("val_loss")))
    assert np.all(np.isfinite(history.column("val_metric")))
    assert np.all(np.diff(history.column("elapsed_s")) >= 0)


def test_logs_sin_validacion_ni_metrica():
    X, y = linear_data(10)
    history = linear_net().fit(X, y, GD(lr=0.1), MSE(), epochs=2)
    assert set(history.records[0]) == {"epoch", "train_loss", "elapsed_s"}
    assert np.all(np.isnan(history.column("val_loss")))


def test_la_perdida_registrada_es_sobre_el_conjunto_completo():
    X, y = linear_data(30)
    net = linear_net()
    history = net.fit(X, y, GD(lr=0.05), MSE(), epochs=4, batch_size=7)
    # Con los pesos finales, la pérdida sobre todo X coincide con la última registrada.
    assert history.column("train_loss")[-1] == pytest.approx(net.evaluate(X, y, MSE())["loss"])


def test_evaluate_no_modifica_los_pesos():
    X, y = linear_data(10)
    net = linear_net()
    before = [p.copy() for p in net.params]
    result = net.evaluate(X, y, MSE(), metric=mean_abs_error)
    assert set(result) == {"loss", "metric"}
    for a, b in zip(net.params, before, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "kwargs",
    [{"epochs": 0}, {"epochs": 5, "batch_size": 0}],
    ids=["epochs_0", "batch_size_0"],
)
def test_parametros_invalidos_fallan(kwargs):
    X, y = linear_data(10)
    with pytest.raises(ValueError):
        linear_net().fit(X, y, GD(), MSE(), **kwargs)


def test_x_e_y_con_distinta_cantidad_de_muestras_falla():
    X, y = linear_data(10)
    with pytest.raises(ValueError):
        linear_net().fit(X, y[:9], GD(), MSE(), epochs=1)


# --- Callbacks ---


def test_callbacks_se_llaman_en_orden():
    calls = []

    class Recorder(Callback):
        def on_train_begin(self, network):
            calls.append("begin")

        def on_epoch_end(self, epoch, logs, network):
            calls.append(epoch)
            return False

        def on_train_end(self, network):
            calls.append("end")

    X, y = linear_data(10)
    linear_net().fit(X, y, GD(), MSE(), epochs=3, callbacks=[Recorder()])
    assert calls == ["begin", 1, 2, 3, "end"]


def test_un_callback_que_devuelve_true_corta_y_todos_ven_la_epoca():
    seen = []

    class StopAt2(Callback):
        def on_epoch_end(self, epoch, logs, network):
            return epoch == 2

    class Watcher(Callback):
        def on_epoch_end(self, epoch, logs, network):
            seen.append(epoch)
            return False

    X, y = linear_data(10)
    history = linear_net().fit(X, y, GD(), MSE(), epochs=10, callbacks=[StopAt2(), Watcher()])
    assert len(history) == 2
    assert seen == [1, 2]


def test_print_progress_escribe_la_primera_y_cada_n_epocas():
    lines: list[str] = []
    X, y = linear_data(10)
    linear_net().fit(X, y, GD(), MSE(), epochs=7,
                     callbacks=[PrintProgress(every=3, write=lines.append)])  # fmt: skip
    assert [line.split(":")[0] for line in lines] == ["epoch 1", "epoch 3", "epoch 6"]
    assert "train_loss=" in lines[0]


def test_print_progress_exige_donde_escribir():
    with pytest.raises(TypeError):
        PrintProgress(every=1)


class FakeLosses(Callback):
    """Reemplaza val_loss por una secuencia fija y cambia los pesos cada época.

    Así se sabe exactamente qué época es la mejor y qué pesos tenía la red.
    """

    def __init__(self, values):
        self.values = values

    def on_epoch_end(self, epoch, logs, network):
        logs["val_loss"] = self.values[epoch - 1]
        for p in network.params:
            p[...] = epoch  # los pesos de la época k valen k
        return False


def test_early_stopping_corta_por_paciencia_y_restaura_la_mejor_epoca():
    stopper = EarlyStopping(patience=2, monitor="val_loss")
    X, y = linear_data(10)
    net = linear_net()
    history = net.fit(X, y, GD(), MSE(), epochs=10,
                      callbacks=[FakeLosses([1.0, 0.5, 0.6, 0.7, 0.1]), stopper])  # fmt: skip

    assert len(history) == 4  # sin mejora en las épocas 3 y 4
    assert stopper.best_epoch == 2
    assert stopper.stopped_epoch == 4
    for p in net.params:
        np.testing.assert_array_equal(p, 2.0)


def test_early_stopping_restaura_aunque_no_corte():
    stopper = EarlyStopping(patience=5)
    X, y = linear_data(10)
    net = linear_net()
    net.fit(X, y, GD(), MSE(), epochs=3, callbacks=[FakeLosses([0.3, 0.1, 0.2]), stopper])
    assert stopper.stopped_epoch is None
    for p in net.params:
        np.testing.assert_array_equal(p, 2.0)


def test_early_stopping_sin_restaurar_deja_los_ultimos_pesos():
    stopper = EarlyStopping(patience=2, restore_best=False)
    X, y = linear_data(10)
    net = linear_net()
    net.fit(X, y, GD(), MSE(), epochs=10, callbacks=[FakeLosses([1.0, 0.5, 0.6, 0.7]), stopper])
    for p in net.params:
        np.testing.assert_array_equal(p, 4.0)


def test_early_stopping_modo_max_y_min_delta():
    stopper = EarlyStopping(patience=1, monitor="val_loss", mode="max", min_delta=0.05)
    X, y = linear_data(10)
    history = linear_net().fit(X, y, GD(), MSE(), epochs=10,
                               callbacks=[FakeLosses([0.5, 0.9, 0.92, 0.99]), stopper])  # fmt: skip
    # 0.92 no supera 0.9 + 0.05, así que la época 3 no cuenta como mejora.
    assert stopper.best_epoch == 2
    assert len(history) == 3


def test_early_stopping_restaura_in_place():
    stopper = EarlyStopping(patience=1)
    X, y = linear_data(10)
    net = linear_net()
    W = net.layers[0].W
    net.fit(X, y, GD(), MSE(), epochs=5, callbacks=[FakeLosses([0.1, 0.2]), stopper])
    assert net.layers[0].W is W


def test_early_stopping_corta_si_diverge():
    stopper = EarlyStopping(patience=2)
    X, y = linear_data(10)
    history = linear_net().fit(X, y, GD(), MSE(), epochs=10,
                               callbacks=[FakeLosses([1.0, np.nan, np.nan]), stopper])  # fmt: skip
    assert len(history) == 3
    assert stopper.best_epoch == 1


def test_early_stopping_sin_validacion_falla_con_mensaje_claro():
    X, y = linear_data(10)
    with pytest.raises(ValueError, match="val_loss"):
        linear_net().fit(X, y, GD(), MSE(), epochs=3, callbacks=[EarlyStopping(patience=2)])


@pytest.mark.parametrize(
    "kwargs", [{"patience": 0}, {"patience": 2, "mode": "maximo"}], ids=["patience", "mode"]
)
def test_early_stopping_parametros_invalidos(kwargs):
    with pytest.raises(ValueError):
        EarlyStopping(**kwargs)


# --- AdaptiveEta (04-matematica §4.1) ---


def lr_after_each_epoch(eta: AdaptiveEta, losses, monitor="train_loss") -> list[float]:
    """Pasa una secuencia sintética de pérdidas al callback; devuelve lr tras cada época."""
    eta.on_train_begin(None)
    out = []
    for epoch, value in enumerate(losses, start=1):
        eta.on_epoch_end(epoch, {monitor: value}, None)
        out.append(eta.optimizer.lr)
    return out


def test_adaptive_eta_sube_tras_k_bajas_y_baja_tras_k_prime_subas():
    eta = AdaptiveEta(GD(lr=1.0), a=0.1, b=0.5, k=2, k_prime=2)
    losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.7, 0.8, 0.9, 1.0]
    # Época 3: segunda baja seguida → +a. Reinicia, época 5: otra vez +a.
    # Época 7: segunda suba seguida → ×(1 − b). Reinicia, época 9: otra vez.
    expected = [1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 0.6, 0.6, 0.3]
    np.testing.assert_allclose(lr_after_each_epoch(eta, losses), expected)


def test_adaptive_eta_racha_interrumpida_o_empate_reinicia_contadores():
    eta = AdaptiveEta(GD(lr=1.0), a=0.1, b=0.5, k=3, k_prime=2)
    # Bajas: 1, (suba), 1, 2, (empate), 1, 2 → nunca 3 seguidas.
    # Subas: la única racha es de 1.
    losses = [1.0, 0.9, 1.0, 0.9, 0.8, 0.8, 0.7, 0.6]
    assert lr_after_each_epoch(eta, losses) == [1.0] * len(losses)


def test_adaptive_eta_respeta_min_lr():
    eta = AdaptiveEta(GD(lr=1e-3), a=0.1, b=0.9, k=5, k_prime=1, min_lr=5e-4)
    lrs = lr_after_each_epoch(eta, [1.0, 2.0, 3.0, 4.0])
    assert lrs == [1e-3, 5e-4, 5e-4, 5e-4]


def test_adaptive_eta_monitor_configurable_y_ausente_falla():
    eta = AdaptiveEta(GD(lr=1.0), a=0.5, b=0.5, k=1, k_prime=1, monitor="val_loss")
    assert lr_after_each_epoch(eta, [1.0, 0.5], monitor="val_loss") == [1.0, 1.5]
    with pytest.raises(ValueError, match="val_loss"):
        lr_after_each_epoch(eta, [1.0], monitor="train_loss")


def test_adaptive_eta_funciona_con_cualquier_optimizador():
    opt = Adam(lr=0.01)
    lrs = lr_after_each_epoch(AdaptiveEta(opt, a=0.01, b=0.5, k=1, k_prime=1), [1.0, 0.5, 0.6])
    np.testing.assert_allclose(lrs, [0.01, 0.02, 0.01])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"a": 0.0},
        {"a": -0.1},
        {"b": 0.0},
        {"b": 1.0},
        {"k": 0},
        {"k_prime": 0},
        {"min_lr": -1.0},
    ],
    ids=lambda kw: next(iter(kw)),
)
def test_adaptive_eta_parametros_invalidos(kwargs):
    params = {"a": 0.1, "b": 0.5, "k": 2, "k_prime": 2} | kwargs
    with pytest.raises(ValueError):
        AdaptiveEta(GD(), **params)


# --- History ---


def test_history_to_csv(tmp_path):
    history = History()
    history.append({"epoch": 1, "train_loss": 0.5, "elapsed_s": 0.01})
    history.append({"epoch": 2, "train_loss": 0.25, "val_loss": 0.3, "elapsed_s": 0.02})
    path = tmp_path / "history.csv"
    history.to_csv(path)

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == list(COLUMNS)
    assert rows[0]["val_loss"] == ""
    assert float(rows[1]["val_loss"]) == 0.3


def test_history_clave_desconocida_falla():
    with pytest.raises(ValueError):
        History().append({"epoch": 1, "accuracy": 0.9})


def test_history_columna_desconocida_falla():
    with pytest.raises(ValueError):
        History().column("accuracy")
