"""Ejercicio previo de validación del motor (Etapa 5 del plan).

Cuatro casos con solución conocida: AND con perceptrón escalón, y = x con
perceptrón lineal, y = tanh(x) con perceptrón no lineal, y XOR con multicapa.
Más la iteración de backprop hecha a mano de docs/verificacion_manual.md.

Los casos que dependen de la inicialización se corren sobre varias semillas:
una sola corrida no alcanza para concluir que el motor funciona.
"""

import numpy as np
import pytest

from core.losses import MSE
from core.network import Network
from core.optimizers import GD
from core.perceptron import fit_perceptron

SEEDS = range(20)

# Entradas de AND y XOR: las cuatro esquinas del cuadrado [−1, 1]².
X_LOGIC = np.array([[-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [1.0, 1.0]])
Y_AND = np.array([[-1.0], [-1.0], [-1.0], [1.0]])
Y_XOR = np.array([[1.0], [1.0], [-1.0], [-1.0]])


# --- AND: perceptrón simple escalón ---


@pytest.mark.parametrize("batch_size", [None, 1], ids=["lote_completo", "online"])
def test_and_error_cero_en_menos_de_100_epocas(batch_size):
    for seed in SEEDS:
        net = Network([2, 1], output_activation="step", initializer="uniform",
                      rng=np.random.default_rng(seed))  # fmt: skip
        history = fit_perceptron(net, X_LOGIC, Y_AND, lr=0.1, epochs=99, batch_size=batch_size)

        error = history.column("train_loss")
        assert error[-1] == 0.0, f"semilla {seed}: no llegó a error 0 en 99 épocas"
        np.testing.assert_array_equal(net.predict(X_LOGIC), Y_AND)


def test_perceptron_escalon_no_puede_con_xor():
    # XOR no es linealmente separable: con una recta, al menos una de las
    # cuatro esquinas queda mal, así que el error nunca baja de 1/4.
    for seed in SEEDS:
        net = Network([2, 1], output_activation="step", initializer="uniform",
                      rng=np.random.default_rng(seed))  # fmt: skip
        history = fit_perceptron(net, X_LOGIC, Y_XOR, lr=0.1, epochs=200)
        assert history.column("train_loss").min() >= 0.25


def test_regla_del_perceptron_paso_a_mano():
    # Δw = η (y − ŷ) x. Con w = (0, 0) y b = 0, z = 0 y step(0) = 1.
    # Muestra (−1, −1) con y = −1: ŷ = 1, y − ŷ = −2.
    # Δw = 0.1 · (−2) · (−1, −1) = (0.2, 0.2);  Δb = 0.1 · (−2) = −0.2
    net = Network([2, 1], output_activation="step", rng=np.random.default_rng(0))
    layer = net.layers[0]
    layer.W[...] = 0.0
    layer.b[...] = 0.0
    fit_perceptron(net, np.array([[-1.0, -1.0]]), np.array([[-1.0]]), lr=0.1, epochs=1)
    np.testing.assert_allclose(layer.W, [[0.2], [0.2]])
    np.testing.assert_allclose(layer.b, [[-0.2]])


def test_fit_perceptron_rechaza_redes_y_etiquetas_invalidas():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="step"):
        fit_perceptron(Network([2, 1], output_activation="tanh", rng=rng),
                       X_LOGIC, Y_AND, lr=0.1, epochs=1)  # fmt: skip
    with pytest.raises(ValueError, match="step"):
        fit_perceptron(Network([2, 2, 1], output_activation="step", rng=rng),
                       X_LOGIC, Y_AND, lr=0.1, epochs=1)  # fmt: skip
    with pytest.raises(ValueError, match="−1 o 1"):
        fit_perceptron(Network([2, 1], output_activation="step", rng=rng),
                       X_LOGIC, (Y_AND + 1) / 2, lr=0.1, epochs=1)  # fmt: skip


# --- y = x: perceptrón simple lineal ---


def test_lineal_y_igual_x_mse_menor_a_1e4():
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        X = rng.uniform(-1, 1, size=(50, 1))
        net = Network([1, 1], output_activation="identity", rng=rng)
        history = net.fit(X, X.copy(), GD(lr=0.1), MSE(), epochs=300)
        assert history.column("train_loss")[-1] < 1e-4, f"semilla {seed}"


# --- y = tanh(x): perceptrón simple no lineal ---


def test_no_lineal_y_igual_tanh_x_mse_menor_a_1e3():
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        X = rng.uniform(-2, 2, size=(50, 1))
        net = Network([1, 1], output_activation="tanh", rng=rng)
        history = net.fit(X, np.tanh(X), GD(lr=0.5), MSE(), epochs=200)
        assert history.column("train_loss")[-1] < 1e-3, f"semilla {seed}"


# --- XOR: perceptrón multicapa ---


def train_xor(layers: list[int], seed: int) -> Network:
    net = Network(layers, hidden_activation="tanh", output_activation="tanh",
                  rng=np.random.default_rng(seed))  # fmt: skip
    net.fit(X_LOGIC, Y_XOR, GD(lr=0.1), MSE(), epochs=2000)
    return net


def solves_xor(net: Network) -> bool:
    return bool(np.all(np.sign(net.predict(X_LOGIC)) == Y_XOR))


def test_xor_2_3_2_1_resuelve_con_todas_las_semillas():
    failed = [s for s in SEEDS if not solves_xor(train_xor([2, 3, 2, 1], s))]
    assert failed == []


def test_xor_2_2_1_resuelve_con_la_mayoria_de_las_semillas():
    # [2, 2, 1] tiene la capacidad justa para XOR y a veces queda en un mínimo
    # local (ver test siguiente). Con estas 20 semillas resuelve 17; se exige
    # 15 para detectar si algo empeora sin atarse al número exacto.
    solved = [s for s in SEEDS if solves_xor(train_xor([2, 2, 1], s))]
    assert len(solved) >= 15, f"resolvió solo {len(solved)} de {len(SEEDS)}"


def test_xor_2_2_1_minimo_local_conocido():
    # La semilla 2 cae en el mínimo local clásico de XOR: dos muestras bien
    # clasificadas y las otras dos con salida ≈ 0, pérdida ≈ (0 + 0 + 1 + 1) / 4
    # = 0.5 y gradiente ≈ 0. No es un bug del backprop (el gradient check pasa):
    # es un punto donde el descenso por gradiente se queda.
    net = train_xor([2, 2, 1], seed=2)
    y_pred = net.predict(X_LOGIC)
    assert MSE().value(Y_XOR, y_pred) == pytest.approx(0.5, abs=0.01)
    assert np.sum(np.abs(y_pred) < 0.01) == 2
    assert max(np.abs(g).max() for g in net.grads) < 1e-2


# --- Iteración de backprop hecha a mano (docs/verificacion_manual.md) ---


def test_verificacion_manual_221():
    """Reproduce, con el motor, los números calculados a mano en docs/verificacion_manual.md."""
    net = Network([2, 2, 1], hidden_activation="tanh", output_activation="tanh",
                  rng=np.random.default_rng(0))  # fmt: skip
    hidden, output = net.layers
    hidden.W = np.array([[0.5, -0.5], [0.25, 1.0]])
    hidden.b = np.array([[0.0, 0.5]])
    output.W = np.array([[2.0], [1.0]])
    output.b = np.array([[0.1]])
    X = np.array([[1.0, -1.0]])
    y = np.array([[1.0]])
    loss = MSE()

    # Paso 1: forward
    y_pred = net.predict(X)
    np.testing.assert_allclose(hidden.z, [[0.25, -1.0]], atol=1e-6)
    np.testing.assert_allclose(output.z, [[-0.171757]], atol=1e-6)
    np.testing.assert_allclose(y_pred, [[-0.170088]], atol=1e-6)
    assert loss.value(y, y_pred) == pytest.approx(1.369105, abs=1e-6)

    # Paso 2: backward
    net.backward(y, y_pred, loss)
    np.testing.assert_allclose(output.grad_W, [[-0.556571], [1.730703]], atol=1e-6)
    np.testing.assert_allclose(output.grad_b, [[-2.272474]], atol=1e-6)
    np.testing.assert_allclose(hidden.grad_W, [[-4.272319, -0.954381], [4.272319, 0.954381]],
                               atol=1e-6)  # fmt: skip
    np.testing.assert_allclose(hidden.grad_b, [[-4.272319, -0.954381]], atol=1e-6)

    # Paso 3: actualización con η = 0.1
    GD(lr=0.1).step(net.params, net.grads)
    np.testing.assert_allclose(hidden.W, [[0.927232, -0.404562], [-0.177232, 0.904562]],
                               atol=1e-6)  # fmt: skip
    np.testing.assert_allclose(hidden.b, [[0.427232, 0.595438]], atol=1e-6)
    np.testing.assert_allclose(output.W, [[2.055657], [0.826930]], atol=1e-6)
    np.testing.assert_allclose(output.b, [[0.327247]], atol=1e-6)

    # Paso 4: la pérdida bajó
    assert loss.value(y, net.predict(X)) == pytest.approx(0.004295, abs=1e-6)
