# TP3 SIA — Perceptrón Simple y Multicapa

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest                              # suite completa
ruff check . && ruff format --check .
```

## CI/CD

Por cada PR o push se corre un script que clona el repo, instala las dependencias de `requirements.txt` y corre `ruff check` y `pytest`.

Convenciones y reglas del proyecto: ver `CLAUDE.md`. Plan por etapas: ver `docs/PLAN_MOTOR.md`.

---

## Estado del motor

| Etapa | Qué | Estado |
| --- | --- | --- |
| 0 | Andamiaje: estructura, dependencias, pytest, CI | ✅ |
| 1 | Activaciones, pérdidas e inicializadores | ✅ |
| 2 | Capa `Dense` y forward de `Network` | pendiente |
| 3 | Backward y gradient check numérico | pendiente |
| 4 en adelante | `fit`, validación, runner, optimizadores, análisis | pendiente |

### Qué hay en `core/`

Todo array tiene forma `(n_muestras, n_features)`. Cada módulo expone objetos (no funciones sueltas) y un registro por nombre para que la config JSON pueda referirse a ellos con un string. Un nombre o parámetro desconocido lanza error, nunca se ignora.

**`activations.py`** — cada activación tiene `forward(z)` y `backward(z)`, donde `backward` es la derivada evaluada en `z` (la pre-activación, no la salida).

| Nombre | Clase | Parámetros | Notas |
| --- | --- | --- | --- |
| `step` | `Step` | — | Devuelve ±1. `backward` lanza error: se entrena con la regla del perceptrón, no con backprop. |
| `identity` | `Identity` | — | |
| `tanh` | `Tanh` | `beta=1.0` | `tanh(beta·z)`, imagen (−1, 1). |
| `sigmoid` | `Sigmoid` | `beta=1.0` | Imagen (0, 1). Implementación estable, no desborda con `z` grande. |
| `relu` | `ReLU` | — | Por convención, derivada 0 en `z = 0`. |
| `softmax` | `Softmax` | — | Por fila, estable (resta el máximo). `backward` lanza error: ver decisión 2. |

Uso: `get_activation("tanh", beta=2.0)`.

**`losses.py`** — cada pérdida tiene `value(y_true, y_pred) -> float` y `grad(y_true, y_pred) -> ndarray` (dL/dŷ, misma forma que `y_pred`).

| Nombre | Clase | Salida esperada |
| --- | --- | --- |
| `mse` | `MSE` | cualquiera |
| `binary_crossentropy` | `BinaryCrossEntropy` | `sigmoid`, con `y ∈ {0, 1}` |
| `categorical_crossentropy` | `CategoricalCrossEntropy` | `softmax`, con `y` one-hot |

Uso: `get_loss("mse")`.

**`initializers.py`** — cada inicializador se llama como `init(n_in, n_out, rng)` y devuelve `W` de forma `(n_in, n_out)`. Los bias no pasan por acá: arrancan en cero.

| Nombre | Clase | Distribución |
| --- | --- | --- |
| `uniform` | `Uniform` | `U(low, high)`, default `U(−0.5, 0.5)` |
| `xavier` | `Xavier` | `U(−r, r)` con `r = √(6 / (n_in + n_out))` |
| `he` | `He` | `N(0, 2 / n_in)` |

Uso: `get_initializer("uniform", low=-1.0, high=1.0)`.

### Qué verifican los tests

- `test_activations.py`: cada derivada analítica contra la numérica `(f(z+ε) − f(z−ε)) / 2ε`, con ε = 1e-5 y tolerancia 1e-7, sobre valores negativos, cero y grandes. Estabilidad de `Sigmoid` y `Softmax`.
- `test_losses.py`: el mismo chequeo numérico para el gradiente de cada pérdida; que el atajo de Softmax + entropía cruzada coincida con la derivada numérica respecto de `z`; valores calculados a mano; que no aparezcan `inf` ni `nan` con probabilidades 0 o 1.
- `test_initializers.py`: formas, límites, varianza de He, y que la misma semilla dé los mismos pesos sin depender del estado global de NumPy.

---

## Decisiones de diseño a validar con el equipo

Estas decisiones no estaban fijadas en `CLAUDE.md` ni en el plan. Si alguien no está de acuerdo, conviene cambiarlas **antes de la Etapa 3**, porque el gradient check y el learning rate dependen de ellas.

### 1. Las pérdidas se promedian, y el gradiente ya viene dividido por `n`

| Pérdida | Fórmula | Gradiente `grad` |
| --- | --- | --- |
| MSE | `mean((y − ŷ)²)` sobre **todas** las muestras y salidas | `2 (ŷ − y) / (n · m)` |
| Entropía cruzada binaria | `−mean(y log ŷ + (1−y) log(1−ŷ))` sobre **todas** las muestras y salidas | `(ŷ − y) / (ŷ (1 − ŷ)) / (n · m)` |
| Entropía cruzada categórica | `−(1/n) Σ_muestras Σ_clases y log ŷ`: **suma** sobre clases, **promedia** sobre muestras | `−y / ŷ / n` |

(`n` = muestras del lote, `m` = salidas.)

Consecuencias:

- La capa **solo suma** sobre el lote (`x.T @ delta`); no vuelve a dividir por `n`. Si alguien divide de nuevo, el gradiente queda `n` veces más chico y el gradient check lo detecta.
- El valor de la pérdida no depende del tamaño del lote, así que el learning rate se puede comparar entre corridas con distinto `batch_size`.
- La MSE usa el factor 1, no el ½ de algunas bibliotecas: el gradiente tiene un 2 adelante. Con una sola salida, el valor coincide con el MSE que reporta la métrica.

### 2. Softmax + entropía cruzada categórica se resuelven juntas, con `softmax_delta`

La derivada de Softmax es una matriz jacobiana por muestra, no un array de la forma de `z`, así que `Softmax.backward` lanza error a propósito. En su lugar, `CategoricalCrossEntropy` tiene un método extra:

```python
CategoricalCrossEntropy().softmax_delta(y_true, y_pred)  # dL/dz = (ŷ − y) / n
```

**Contrato para la Etapa 3:** cuando la capa de salida es `Softmax`, la red calcula el delta de salida con `loss.softmax_delta(...)` y **no** multiplica por la derivada de la activación. En cualquier otro caso usa `loss.grad(...) * activation.backward(z)`. Además de ser más barato, este atajo es más estable porque no divide por ŷ.

Softmax con otra pérdida (por ejemplo, MSE) no está soportado.

### 3. Variantes de inicialización

- **Xavier** es la versión **uniforme** (Glorot uniforme). Es la recomendada para capas `tanh` y `sigmoid`.
- **He** es la versión **normal**. Es la recomendada para capas `relu`.
- **Uniform** usa por default `U(−0.5, 0.5)`, la inicialización clásica del perceptrón simple.
- Los **bias arrancan en cero**; el inicializador solo genera `W`.

### 4. Detalles de robustez

- **Formas estrictas en las pérdidas.** Si `y_true` e `y_pred` tienen formas distintas, se lanza `ValueError`. Sin este chequeo, `(n,)` contra `(n, 1)` se expande a `(n, n)` y la pérdida da un número creíble pero incorrecto. Los targets siempre se pasan como `(n, 1)`, nunca como `(n,)`.
- **Recorte de probabilidades.** Las entropías cruzadas recortan ŷ a `[1e-12, 1 − 1e-12]` antes del log, así la pérdida no se vuelve infinita cuando la salida satura.
