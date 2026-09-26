# Verificación manual: una iteración de backprop en `[2, 2, 1]`

Una iteración completa (forward, pérdida, backward y actualización), calculada fórmula por fórmula con escalares y comparada contra el motor. El test `tests/test_validation.py::test_verificacion_manual_221` corre la misma iteración con el motor y verifica cada número de este documento con tolerancia 1e-6.

> **Pendiente para el equipo:** estos números se calcularon con las fórmulas escalares de abajo, escritas una por una sin usar el motor (sin matrices, `Network` ni `Dense`). El plan pide que la iteración se haga **a mano**, y sirve para la defensa oral: conviene que alguien la rehaga en papel con calculadora y confirme que llega a los mismos valores.

Los números se muestran redondeados a 6 decimales, pero las cuentas se hicieron sin redondear en los pasos intermedios.

---

## Configuración

- **Red:** 2 entradas → 2 neuronas ocultas `tanh` → 1 salida `tanh`. Es la red del XOR.
- **Pérdida:** MSE. Con una muestra y una salida, `L = (y − ŷ)²`.
- **Optimizador:** GD con `η = 0.1`.
- **Muestra:** `x = (1, −1)`, `y = 1`.

**Pesos iniciales.** `wᵢⱼ` va de la entrada `i` a la neurona oculta `j`; `vⱼ` va de la oculta `j` a la salida.

| Capa | Parámetro | Valor |
| --- | --- | --- |
| Oculta | `w₁₁`, `w₁₂` (desde x₁) | 0.5, −0.5 |
| Oculta | `w₂₁`, `w₂₂` (desde x₂) | 0.25, 1.0 |
| Oculta | `b₁`, `b₂` | 0.0, 0.5 |
| Salida | `v₁`, `v₂` | 2.0, 1.0 |
| Salida | `c` | 0.1 |

En el motor: `W1 = [[w₁₁, w₁₂], [w₂₁, w₂₂]]` de forma `(2, 2)`, `b1 = [[b₁, b₂]]`, `W2 = [[v₁], [v₂]]` de forma `(2, 1)` y `b2 = [[c]]`.

---

## Paso 1 — Forward

**Capa oculta:**

| | Fórmula | Valor |
| --- | --- | --- |
| `z₁` | `x₁·w₁₁ + x₂·w₂₁ + b₁ = 1·0.5 + (−1)·0.25 + 0` | 0.250000 |
| `z₂` | `x₁·w₁₂ + x₂·w₂₂ + b₂ = 1·(−0.5) + (−1)·1.0 + 0.5` | −1.000000 |
| `h₁` | `tanh(z₁)` | 0.244919 |
| `h₂` | `tanh(z₂)` | −0.761594 |

**Capa de salida:**

| | Fórmula | Valor |
| --- | --- | --- |
| `z_o` | `h₁·v₁ + h₂·v₂ + c = 0.244919·2 + (−0.761594)·1 + 0.1` | −0.171757 |
| `ŷ` | `tanh(z_o)` | −0.170088 |

**Pérdida:** `L = (y − ŷ)² = (1 − (−0.170088))² = 1.170088²` = **1.369105**

---

## Paso 2 — Backward

Se usa `tanh'(z) = 1 − tanh(z)²`, evaluada en la pre-activación `z`, que se escribe con la salida ya calculada.

**Delta de salida:**

| | Fórmula | Valor |
| --- | --- | --- |
| `∂L/∂ŷ` | `2(ŷ − y) = 2(−0.170088 − 1)` | −2.340175 |
| `tanh'(z_o)` | `1 − ŷ² = 1 − 0.170088²` | 0.971070 |
| `δ_o` | `∂L/∂ŷ · tanh'(z_o)` | −2.272474 |

**Gradientes de la capa de salida:**

| | Fórmula | Valor |
| --- | --- | --- |
| `∂L/∂v₁` | `δ_o · h₁ = −2.272474 · 0.244919` | −0.556571 |
| `∂L/∂v₂` | `δ_o · h₂ = −2.272474 · (−0.761594)` | 1.730703 |
| `∂L/∂c` | `δ_o` | −2.272474 |

**Deltas de la capa oculta.** El error vuelve por el peso que conecta cada neurona con la salida:

| | Fórmula | Valor |
| --- | --- | --- |
| `tanh'(z₁)` | `1 − h₁² = 1 − 0.244919²` | 0.940015 |
| `tanh'(z₂)` | `1 − h₂² = 1 − 0.761594²` | 0.419974 |
| `δ₁` | `δ_o · v₁ · tanh'(z₁) = −2.272474 · 2 · 0.940015` | −4.272319 |
| `δ₂` | `δ_o · v₂ · tanh'(z₂) = −2.272474 · 1 · 0.419974` | −0.954381 |

**Gradientes de la capa oculta.** Como `x₂ = −1`, los gradientes de los pesos que salen de `x₂` son los de `x₁` con el signo cambiado:

| | Fórmula | Valor |
| --- | --- | --- |
| `∂L/∂w₁₁` | `δ₁ · x₁` | −4.272319 |
| `∂L/∂w₁₂` | `δ₂ · x₁` | −0.954381 |
| `∂L/∂w₂₁` | `δ₁ · x₂` | 4.272319 |
| `∂L/∂w₂₂` | `δ₂ · x₂` | 0.954381 |
| `∂L/∂b₁` | `δ₁` | −4.272319 |
| `∂L/∂b₂` | `δ₂` | −0.954381 |

---

## Paso 3 — Actualización (`θ ← θ − η · ∂L/∂θ`, `η = 0.1`)

| Parámetro | Antes | Gradiente | Después |
| --- | --- | --- | --- |
| `w₁₁` | 0.5 | −4.272319 | 0.927232 |
| `w₁₂` | −0.5 | −0.954381 | −0.404562 |
| `w₂₁` | 0.25 | 4.272319 | −0.177232 |
| `w₂₂` | 1.0 | 0.954381 | 0.904562 |
| `b₁` | 0.0 | −4.272319 | 0.427232 |
| `b₂` | 0.5 | −0.954381 | 0.595438 |
| `v₁` | 2.0 | −0.556571 | 2.055657 |
| `v₂` | 1.0 | 1.730703 | 0.826930 |
| `c` | 0.1 | −2.272474 | 0.327247 |

---

## Paso 4 — La pérdida bajó

Repitiendo el forward con los pesos nuevos, `ŷ = 0.934461` y `L = (1 − 0.934461)²` = **0.004295**. Antes era 1.369105.

La caída es tan grande porque hay una sola muestra y `η = 0.1` ya es un paso grande para este problema. El objetivo acá es verificar el signo y la magnitud de cada gradiente, no la velocidad de entrenamiento.

---

## Qué confirma esta verificación

- **Signos:** todos los gradientes empujan en la dirección que reduce la pérdida, y la pérdida baja después del paso.
- **Derivada evaluada en `z`:** `tanh'` se calculó en la pre-activación, escrita con la salida como `1 − tanh(z)²`.
- **Retropropagación por los pesos:** `δ₁` usa `v₁` y `δ₂` usa `v₂`. Esa es la transpuesta `δ @ W.T` de `Dense.backward_z`.
- **Gradiente de un peso = delta × entrada:** es la transpuesta `x.T @ δ`. Se ve en el cambio de signo de los pesos que salen de `x₂ = −1`.
- **Bias:** su gradiente es el delta sin multiplicar por ninguna entrada.

## Cómo reproducirlo con el motor

```bash
pytest tests/test_validation.py::test_verificacion_manual_221 -v
```

O en una consola de Python, para ver los valores:

```python
import numpy as np
from core.losses import MSE
from core.network import Network
from core.optimizers import GD

net = Network(
    [2, 2, 1], hidden_activation="tanh", output_activation="tanh", rng=np.random.default_rng(0)
)
hidden, output = net.layers
hidden.W, hidden.b = np.array([[0.5, -0.5], [0.25, 1.0]]), np.array([[0.0, 0.5]])
output.W, output.b = np.array([[2.0], [1.0]]), np.array([[0.1]])

X, y = np.array([[1.0, -1.0]]), np.array([[1.0]])
y_pred = net.predict(X)
print("ŷ =", y_pred, " L =", MSE().value(y, y_pred))
net.backward(y, y_pred, MSE())
for name, g in zip(["dW1", "db1", "dW2", "db2"], net.grads):
    print(name, g)
GD(lr=0.1).step(net.params, net.grads)
print("L después =", MSE().value(y, net.predict(X)))
```
