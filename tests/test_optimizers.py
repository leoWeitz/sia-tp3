import numpy as np
import pytest

from core.losses import MSE
from core.network import Network
from core.optimizers import GD, OPTIMIZERS, get_optimizer


def test_gd_paso_calculado_a_mano():
    # θ ← θ − lr · g = (1, −2) − 0.1 · (4, −6) = (0.6, −1.4)
    theta = np.array([[1.0, -2.0]])
    GD(lr=0.1).step([theta], [np.array([[4.0, -6.0]])])
    np.testing.assert_allclose(theta, [[0.6, -1.4]])


def test_gd_actualiza_los_pesos_de_la_red_in_place():
    net = Network([2, 3, 1], rng=np.random.default_rng(0))
    W_before = net.layers[0].W
    copy_before = W_before.copy()

    X = np.random.default_rng(1).normal(size=(4, 2))
    y = np.zeros((4, 1))
    net.backward(y, net.predict(X), MSE())
    GD(lr=0.1).step(net.params, net.grads)

    assert net.layers[0].W is W_before  # mismo objeto, no una copia
    assert not np.array_equal(net.layers[0].W, copy_before)


def test_gd_baja_la_perdida_en_y_igual_a_x():
    rng = np.random.default_rng(0)
    net = Network([1, 1], output_activation="identity", rng=rng)
    X = rng.uniform(-1, 1, size=(50, 1))
    y = X.copy()
    loss, opt = MSE(), GD(lr=0.1)

    losses = []
    for _ in range(100):
        y_pred = net.predict(X)
        losses.append(loss.value(y, y_pred))
        net.backward(y, y_pred, loss)
        opt.step(net.params, net.grads)

    assert all(b <= a for a, b in zip(losses, losses[1:], strict=False))
    assert losses[-1] < 1e-2 * losses[0]


def test_gd_listas_de_distinto_largo_fallan():
    with pytest.raises(ValueError):
        GD().step([np.zeros(2), np.zeros(2)], [np.zeros(2)])


@pytest.mark.parametrize("lr", [0.0, -0.1])
def test_gd_lr_no_positivo_falla(lr):
    with pytest.raises(ValueError):
        GD(lr=lr)


def test_registro_construye_cada_optimizador():
    for name, cls in OPTIMIZERS.items():
        assert isinstance(get_optimizer(name), cls)


def test_registro_pasa_parametros():
    assert get_optimizer("gd", lr=0.5).lr == 0.5


def test_registro_nombre_desconocido_falla():
    with pytest.raises(ValueError):
        get_optimizer("sgd")
