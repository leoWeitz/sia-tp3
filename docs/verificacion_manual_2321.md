# Verificación manual: una iteración de backprop en `[2, 3, 2, 1]`

Una iteración completa (forward, pérdida, backward y actualización), calculada fórmula por fórmula con escalares y comparada contra el motor. Es la segunda arquitectura de XOR del enunciado y complementa [`verificacion_manual.md`](verificacion_manual.md) (`[2, 2, 1]`): acá el error se retropropaga a través de **dos** capas ocultas. El test `tests/test_validation.py::test_verificacion_manual_2321` corre la misma iteración con el motor y verifica cada número de este documento con tolerancia 1e-6.

> **Pendiente para el equipo:** estos números se calcularon con las fórmulas escalares de abajo, escritas una por una sin usar el motor (sin matrices, `Network` ni `Dense`). Hay que rehacerlos en papel con calculadora y confirmar que se llega a los mismos valores.

Los números se muestran redondeados a 6 decimales, pero las cuentas se hicieron sin redondear en los pasos intermedios.

---

## Configuración

- **Red:** 2 entradas → 3 neuronas ocultas `tanh` → 2 neuronas ocultas `tanh` → 1 salida `tanh`.
- **Pérdida:** MSE. Con una muestra y una salida, `L = (y − ŷ)²`.
- **Optimizador:** GD con `η = 0.1`.
- **Muestra:** `x = (1, 1)`, `y = −1` (una esquina de XOR).

**Notación.** El superíndice es la capa. `w¹ᵢⱼ` va de la entrada `i` a la neurona `j` de la primera capa oculta; `w²ⱼₖ` va de la neurona `j` de la primera oculta a la neurona `k` de la segunda; `vₖ` va de la neurona `k` de la segunda oculta a la salida. `b¹ⱼ`, `b²ₖ` y `c` son los bias. En cada neurona, `z` es la pre-activación y `h = tanh(z)` la salida.

**Pesos iniciales:**

| Capa | Parámetro | Valor |
| --- | --- | --- |
| Oculta 1 | `w¹₁₁`, `w¹₁₂`, `w¹₁₃` (desde x₁) | 0.5, −0.5, 0.25 |
| Oculta 1 | `w¹₂₁`, `w¹₂₂`, `w¹₂₃` (desde x₂) | 0.25, 0.5, −0.5 |
| Oculta 1 | `b¹₁`, `b¹₂`, `b¹₃` | 0.0, 0.25, −0.25 |
| Oculta 2 | `w²₁₁`, `w²₁₂` (desde h¹₁) | 0.5, −1.0 |
| Oculta 2 | `w²₂₁`, `w²₂₂` (desde h¹₂) | −0.5, 0.5 |
| Oculta 2 | `w²₃₁`, `w²₃₂` (desde h¹₃) | 1.0, 0.25 |
| Oculta 2 | `b²₁`, `b²₂` | 0.1, −0.1 |
| Salida | `v₁`, `v₂` (desde h²₁, h²₂) | 1.0, −0.5 |
| Salida | `c` | 0.2 |

En el motor: `W1 = [[w¹₁₁, w¹₁₂, w¹₁₃], [w¹₂₁, w¹₂₂, w¹₂₃]]` de forma `(2, 3)`, `b1 = [[b¹₁, b¹₂, b¹₃]]`, `W2 = [[w²₁₁, w²₁₂], [w²₂₁, w²₂₂], [w²₃₁, w²₃₂]]` de forma `(3, 2)`, `b2 = [[b²₁, b²₂]]`, `W3 = [[v₁], [v₂]]` de forma `(2, 1)` y `b3 = [[c]]`.

---

## Paso 1 — Forward

**Capa oculta 1** (`z¹ⱼ = x₁·w¹₁ⱼ + x₂·w¹₂ⱼ + b¹ⱼ`, con `x₁ = x₂ = 1`):

| | Fórmula | Valor |
| --- | --- | --- |
| `z¹₁` | `1·0.5 + 1·0.25 + 0.0` | 0.750000 |
| `z¹₂` | `1·(−0.5) + 1·0.5 + 0.25` | 0.250000 |
| `z¹₃` | `1·0.25 + 1·(−0.5) + (−0.25)` | −0.500000 |
| `h¹₁` | `tanh(z¹₁)` | 0.635149 |
| `h¹₂` | `tanh(z¹₂)` | 0.244919 |
| `h¹₃` | `tanh(z¹₃)` | −0.462117 |

**Capa oculta 2** (`z²ₖ = h¹₁·w²₁ₖ + h¹₂·w²₂ₖ + h¹₃·w²₃ₖ + b²ₖ`):

| | Fórmula | Valor |
| --- | --- | --- |
| `z²₁` | `0.635149·0.5 + 0.244919·(−0.5) + (−0.462117)·1.0 + 0.1` | −0.167002 |
| `z²₂` | `0.635149·(−1.0) + 0.244919·0.5 + (−0.462117)·0.25 + (−0.1)` | −0.728219 |
| `h²₁` | `tanh(z²₁)` | −0.165467 |
| `h²₂` | `tanh(z²₂)` | −0.621974 |

**Capa de salida:**

| | Fórmula | Valor |
| --- | --- | --- |
| `z_o` | `h²₁·v₁ + h²₂·v₂ + c = (−0.165467)·1.0 + (−0.621974)·(−0.5) + 0.2` | 0.345521 |
| `ŷ` | `tanh(z_o)` | 0.332397 |

**Pérdida:** `L = (y − ŷ)² = (−1 − 0.332397)² = (−1.332397)²` = **1.775282**

---

## Paso 2 — Backward

Se usa `tanh'(z) = 1 − tanh(z)²`, evaluada en la pre-activación `z`, que se escribe con la salida ya calculada.

**Delta de salida:**

| | Fórmula | Valor |
| --- | --- | --- |
| `∂L/∂ŷ` | `2(ŷ − y) = 2(0.332397 − (−1))` | 2.664794 |
| `tanh'(z_o)` | `1 − ŷ² = 1 − 0.332397²` | 0.889512 |
| `δ_o` | `∂L/∂ŷ · tanh'(z_o)` | 2.370367 |

**Gradientes de la capa de salida:**

| | Fórmula | Valor |
| --- | --- | --- |
| `∂L/∂v₁` | `δ_o · h²₁ = 2.370367 · (−0.165467)` | −0.392217 |
| `∂L/∂v₂` | `δ_o · h²₂ = 2.370367 · (−0.621974)` | −1.474308 |
| `∂L/∂c` | `δ_o` | 2.370367 |

**Deltas de la capa oculta 2.** El error vuelve por el peso que conecta cada neurona con la salida:

| | Fórmula | Valor |
| --- | --- | --- |
| `tanh'(z²₁)` | `1 − (h²₁)² = 1 − (−0.165467)²` | 0.972621 |
| `tanh'(z²₂)` | `1 − (h²₂)² = 1 − (−0.621974)²` | 0.613148 |
| `δ²₁` | `δ_o · v₁ · tanh'(z²₁) = 2.370367 · 1.0 · 0.972621` | 2.305468 |
| `δ²₂` | `δ_o · v₂ · tanh'(z²₂) = 2.370367 · (−0.5) · 0.613148` | −0.726693 |

**Gradientes de la capa oculta 2.** Cada peso es el delta de la neurona a la que llega por la salida de la neurona de la que sale:

| | Fórmula | Valor |
| --- | --- | --- |
| `∂L/∂w²₁₁` | `δ²₁ · h¹₁ = 2.305468 · 0.635149` | 1.464316 |
| `∂L/∂w²₁₂` | `δ²₂ · h¹₁ = −0.726693 · 0.635149` | −0.461558 |
| `∂L/∂w²₂₁` | `δ²₁ · h¹₂ = 2.305468 · 0.244919` | 0.564652 |
| `∂L/∂w²₂₂` | `δ²₂ · h¹₂ = −0.726693 · 0.244919` | −0.177981 |
| `∂L/∂w²₃₁` | `δ²₁ · h¹₃ = 2.305468 · (−0.462117)` | −1.065396 |
| `∂L/∂w²₃₂` | `δ²₂ · h¹₃ = −0.726693 · (−0.462117)` | 0.335817 |
| `∂L/∂b²₁` | `δ²₁` | 2.305468 |
| `∂L/∂b²₂` | `δ²₂` | −0.726693 |

**Deltas de la capa oculta 1.** Esto es lo que `[2, 2, 1]` no prueba: cada neurona de la primera oculta está conectada con las **dos** neuronas de la segunda, así que su delta suma el error que vuelve por los dos pesos que salen de ella:

| | Fórmula | Valor |
| --- | --- | --- |
| `tanh'(z¹₁)` | `1 − (h¹₁)² = 1 − 0.635149²` | 0.596586 |
| `tanh'(z¹₂)` | `1 − (h¹₂)² = 1 − 0.244919²` | 0.940015 |
| `tanh'(z¹₃)` | `1 − (h¹₃)² = 1 − (−0.462117)²` | 0.786448 |
| `δ¹₁` | `tanh'(z¹₁) · (w²₁₁·δ²₁ + w²₁₂·δ²₂) = 0.596586 · (0.5·2.305468 + (−1.0)·(−0.726693))` | 1.121239 |
| `δ¹₂` | `tanh'(z¹₂) · (w²₂₁·δ²₁ + w²₂₂·δ²₂) = 0.940015 · ((−0.5)·2.305468 + 0.5·(−0.726693))` | −1.425138 |
| `δ¹₃` | `tanh'(z¹₃) · (w²₃₁·δ²₁ + w²₃₂·δ²₂) = 0.786448 · (1.0·2.305468 + 0.25·(−0.726693))` | 1.670254 |

**Gradientes de la capa oculta 1.** Como `x₁ = x₂ = 1`, los gradientes de los pesos que salen de `x₁` y de `x₂` son iguales, y los dos coinciden con el delta (el cambio de signo por una entrada negativa ya se verificó en `[2, 2, 1]`):

| | Fórmula | Valor |
| --- | --- | --- |
| `∂L/∂w¹₁₁` | `δ¹₁ · x₁` | 1.121239 |
| `∂L/∂w¹₁₂` | `δ¹₂ · x₁` | −1.425138 |
| `∂L/∂w¹₁₃` | `δ¹₃ · x₁` | 1.670254 |
| `∂L/∂w¹₂₁` | `δ¹₁ · x₂` | 1.121239 |
| `∂L/∂w¹₂₂` | `δ¹₂ · x₂` | −1.425138 |
| `∂L/∂w¹₂₃` | `δ¹₃ · x₂` | 1.670254 |
| `∂L/∂b¹₁` | `δ¹₁` | 1.121239 |
| `∂L/∂b¹₂` | `δ¹₂` | −1.425138 |
| `∂L/∂b¹₃` | `δ¹₃` | 1.670254 |

---

## Paso 3 — Actualización (`θ ← θ − η · ∂L/∂θ`, `η = 0.1`)

| Parámetro | Antes | Gradiente | Después |
| --- | --- | --- | --- |
| `w¹₁₁` | 0.5 | 1.121239 | 0.387876 |
| `w¹₁₂` | −0.5 | −1.425138 | −0.357486 |
| `w¹₁₃` | 0.25 | 1.670254 | 0.082975 |
| `w¹₂₁` | 0.25 | 1.121239 | 0.137876 |
| `w¹₂₂` | 0.5 | −1.425138 | 0.642514 |
| `w¹₂₃` | −0.5 | 1.670254 | −0.667025 |
| `b¹₁` | 0.0 | 1.121239 | −0.112124 |
| `b¹₂` | 0.25 | −1.425138 | 0.392514 |
| `b¹₃` | −0.25 | 1.670254 | −0.417025 |
| `w²₁₁` | 0.5 | 1.464316 | 0.353568 |
| `w²₁₂` | −1.0 | −0.461558 | −0.953844 |
| `w²₂₁` | −0.5 | 0.564652 | −0.556465 |
| `w²₂₂` | 0.5 | −0.177981 | 0.517798 |
| `w²₃₁` | 1.0 | −1.065396 | 1.106540 |
| `w²₃₂` | 0.25 | 0.335817 | 0.216418 |
| `b²₁` | 0.1 | 2.305468 | −0.130547 |
| `b²₂` | −0.1 | −0.726693 | −0.027331 |
| `v₁` | 1.0 | −0.392217 | 1.039222 |
| `v₂` | −0.5 | −1.474308 | −0.352569 |
| `c` | 0.2 | 2.370367 | −0.037037 |

---

## Paso 4 — La pérdida bajó

Repitiendo el forward con los pesos nuevos, `ŷ = −0.665015` y `L = (−1 − (−0.665015))²` = **0.112215**. Antes era 1.775282: la salida pasó de 0.332 (lado equivocado) a −0.665 (lado correcto).

Como en `[2, 2, 1]`, la caída es grande porque hay una sola muestra y `η = 0.1` ya es un paso grande para este problema: el objetivo es verificar el signo y la magnitud de cada gradiente, no la velocidad de entrenamiento.

---

## Qué confirma esta verificación

Además de lo que ya confirma [`verificacion_manual.md`](verificacion_manual.md) (derivada evaluada en `z`, bias con el delta sin multiplicar, cambio de signo por una entrada negativa):

- **Retropropagación a través de dos capas ocultas:** `δ¹` depende de `δ²`, que depende de `δ_o`. Un error en la cadena (por ejemplo, usar los pesos ya actualizados de una capa para calcular el delta de la anterior) cambiaría `δ¹`.
- **Suma sobre las neuronas de la capa siguiente:** cada `δ¹ⱼ` suma dos términos, `w²ⱼ₁·δ²₁ + w²ⱼ₂·δ²₂`. Es el producto `δ² @ W2.T` de `Dense.backward_z`, que en `[2, 2, 1]` tenía un solo término.
- **Capas no cuadradas:** `W1` es `(2, 3)` y `W2` es `(3, 2)`. Confundir `W` con su transpuesta daría un error de forma o números distintos.
- **Signos:** la muestra es negativa (`y = −1`) y la salida inicial positiva, así que todos los gradientes empujan `ŷ` hacia abajo y la pérdida baja después del paso.

## Cómo reproducirlo con el motor

```bash
pytest tests/test_validation.py::test_verificacion_manual_2321 -v
```

O en una consola de Python, para ver los valores:

```python
import numpy as np
from core.losses import MSE
from core.network import Network
from core.optimizers import GD

net = Network(
    [2, 3, 2, 1], hidden_activation="tanh", output_activation="tanh", rng=np.random.default_rng(0)
)
hidden1, hidden2, output = net.layers
hidden1.W = np.array([[0.5, -0.5, 0.25], [0.25, 0.5, -0.5]])
hidden1.b = np.array([[0.0, 0.25, -0.25]])
hidden2.W = np.array([[0.5, -1.0], [-0.5, 0.5], [1.0, 0.25]])
hidden2.b = np.array([[0.1, -0.1]])
output.W, output.b = np.array([[1.0], [-0.5]]), np.array([[0.2]])

X, y = np.array([[1.0, 1.0]]), np.array([[-1.0]])
y_pred = net.predict(X)
print("ŷ =", y_pred, " L =", MSE().value(y, y_pred))
net.backward(y, y_pred, MSE())
for name, g in zip(["dW1", "db1", "dW2", "db2", "dW3", "db3"], net.grads):
    print(name, g)
GD(lr=0.1).step(net.params, net.grads)
print("L después =", MSE().value(y, net.predict(X)))
```
