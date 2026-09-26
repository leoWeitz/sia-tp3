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
| 2 | Capa `Dense` y forward de `Network` | ✅ |
| 3 | Backward, gradient check numérico y optimizador `GD` | ✅ |
| 4 | `fit`, mini-batch, `History` y callbacks | ✅ |
| 5 | Ejercicio previo de validación: AND, `y = x`, `y = tanh(x)`, XOR y verificación manual | ✅ (falta rehacer la verificación manual en papel) |
| 6 en adelante | Runner y datos, Momentum y Adam, análisis | pendiente |

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

**`layers.py`** — `Dense(n_in, n_out, activation, initializer, rng)` calcula `a = activation(x @ W + b)`.

- `W` tiene forma `(n_in, n_out)` y `b` forma `(1, n_out)`; el bias se suma por broadcasting.
- `forward(x)` guarda `x` y `z = x @ W + b` en la capa, porque el backward los necesita.
- Si `x` no tiene forma `(n_muestras, n_in)`, lanza error. Un vector suelto `(n_in,)` también se rechaza: una sola muestra se pasa como `(1, n_in)`.
- `backward(grad_a)` recibe dL/da, lo multiplica por la derivada de la activación evaluada en `z` y sigue como `backward_z`.
- `backward_z(delta)` recibe dL/dz ya calculado, guarda `grad_W = x.T @ delta` y `grad_b = delta.sum(axis=0)`, y devuelve dL/dx `= delta @ W.T` para la capa anterior. Es la entrada que usa el par Softmax + CCE (ver decisión 2).
- `params` devuelve `[W, b]` y `grads` devuelve `[grad_W, grad_b]`, en el mismo orden.

**`network.py`** — `Network(layer_sizes, *, hidden_activation, output_activation, initializer, rng)` arma una capa `Dense` por cada par de tamaños consecutivos.

```python
rng = np.random.default_rng(42)

# Perceptrón simple lineal
Network([2, 1], output_activation="identity", rng=rng)

# Perceptrón simple no lineal
Network([2, 1], output_activation=Tanh(beta=2.0), rng=rng)

# Multicapa
Network(
    [784, 64, 10], hidden_activation="relu", output_activation="softmax", initializer="he", rng=rng
)
```

- `predict(X)` encadena los forwards: de `(n_muestras, layer_sizes[0])` a `(n_muestras, layer_sizes[-1])`.
- `backward(y_true, y_pred, loss)` calcula los gradientes de todas las capas, recorriéndolas de la última a la primera. `y_pred` tiene que ser la salida del `predict` inmediatamente anterior.
- `params` y `grads` son las listas `[W1, b1, W2, b2, ...]` y `[grad_W1, grad_b1, ...]`, en el mismo orden, listas para pasarle al optimizador.
- `n_params` da la cantidad total de parámetros entrenables, para el `metrics.json`.

**`optimizers.py`** — un optimizador tiene `step(params, grads)` y actualiza cada parámetro in place. No calcula ni modifica gradientes.

| Nombre | Clase | Parámetros | Paso |
| --- | --- | --- | --- |
| `gd` | `GD` | `lr=0.01` | `θ ← θ − lr · g` |

Uso: `get_optimizer("gd", lr=0.1)`. Momentum y Adam llegan en la Etapa 7.

**Entrenamiento con `fit`** — `Network.fit` repite, época tras época, el paso `predict` → `backward` → `optimizer.step` sobre cada lote:

```python
# X: (n_muestras, n_in), y: (n_muestras, n_out)
history = net.fit(
    X,
    y,
    optimizer=get_optimizer("gd", lr=0.1),
    loss=get_loss("mse"),
    epochs=200,
    batch_size=32,  # None = lote completo, 1 = estocástico
    validation=(X_val, y_val),  # opcional: agrega val_loss y val_metric
    metric=accuracy,  # opcional: f(y_true, y_pred) -> float
    callbacks=[PrintProgress(every=10, write=print), EarlyStopping(patience=20)],
)
history.to_csv("results/<run_id>/history.csv")
```

- **Lotes:** con `batch_size=k`, el orden de las muestras se baraja al empezar cada época y el último lote puede ser más chico. Con lote completo no se baraja, porque el orden no cambia el gradiente.
- **`evaluate(X, y, loss, metric=None)`** devuelve `{"loss": ..., "metric": ...}` sin entrenar.

**`perceptron.py`** — `fit_perceptron(network, X, y, lr, epochs, batch_size=None, callbacks=(), rng=None)` entrena el perceptrón simple escalón con la regla del perceptrón, `Δw = η (y − ŷ) x`, porque el escalón no es derivable.

```python
net = Network([2, 1], output_activation="step", initializer="uniform", rng=rng)
history = fit_perceptron(net, X, y, lr=0.1, epochs=100)  # y: (n_muestras, 1) con valores ±1
```

- La red tiene que tener una sola capa con salida `step`, y las etiquetas tienen que ser −1 o 1; si no, lanza error.
- En el historial, `train_loss` es la **tasa de error** (fracción de muestras mal clasificadas), no una pérdida continua.
- Devuelve un `History` y acepta los mismos callbacks que `fit`.

**`history.py`** — `History` guarda un dict por época con las columnas del `history.csv` del contrato de resultados: `epoch, train_loss, val_loss, train_metric, val_metric, elapsed_s`.

- `history.column("train_loss")` devuelve los valores por época como array, con NaN donde no se registró (por ejemplo, `val_loss` sin validación).
- `history.to_csv(path)` escribe el CSV; lo que no se registró queda vacío.

**`callbacks.py`** — la única forma de observar o cortar un `fit`. Un callback hereda de `Callback` y redefine lo que necesita: `on_train_begin(network)`, `on_epoch_end(epoch, logs, network)` (si devuelve `True`, corta) y `on_train_end(network)`.

| Callback | Qué hace |
| --- | --- |
| `PrintProgress(every, write)` | Reporta los logs en la primera época y cada `every`. `write` es obligatorio, por ejemplo `write=print`. |
| `EarlyStopping(patience, monitor="val_loss", mode="min", min_delta=0.0, restore_best=True)` | Corta si `monitor` no mejora durante `patience` épocas seguidas. Con `restore_best`, al terminar deja la red con los pesos de la mejor época. Expone `best_epoch` y `stopped_epoch`. |

### Qué verifican los tests

- `test_activations.py`: cada derivada analítica contra la numérica `(f(z+ε) − f(z−ε)) / 2ε`, con ε = 1e-5 y tolerancia 1e-7, sobre valores negativos, cero y grandes. Estabilidad de `Sigmoid` y `Softmax`.
- `test_losses.py`: el mismo chequeo numérico para el gradiente de cada pérdida; que el atajo de Softmax + entropía cruzada coincida con la derivada numérica respecto de `z`; valores calculados a mano; que no aparezcan `inf` ni `nan` con probabilidades 0 o 1.
- `test_initializers.py`: formas, límites, varianza de He, y que la misma semilla dé los mismos pesos sin depender del estado global de NumPy.
- `test_forward.py`: el criterio de aceptación de la Etapa 2, una red `[2, 2, 1]` con pesos elegidos a mano cuya salida está calculada paso a paso en los comentarios del test. También verifica que procesar un lote dé lo mismo que procesar cada muestra por separado, la cache de `x` y `z`, las formas de los parámetros y la reproducibilidad por semilla.
- `test_gradients.py`: **el gradient check numérico**, criterio de aceptación de la Etapa 3. Para cada parámetro compara el gradiente del backward contra `(L(θ+ε) − L(θ−ε)) / 2ε` y exige un error relativo menor a 1e-6. Cubre:
  - las 6 combinaciones del plan: capas ocultas `tanh`, `sigmoid` y `relu`, cada una con salida `identity` + MSE y con `softmax` + CCE;
  - salidas `tanh` y `sigmoid` con MSE, y `sigmoid` con BCE;
  - perceptrón simple (sin capas ocultas) y redes con varias capas ocultas.

  Además, **verifica que el propio chequeo detecte bugs**: introduce a propósito cuatro errores típicos de backprop (derivada evaluada en `a` en vez de `z`, bias que no suma sobre el lote, signo invertido, falta de la derivada de la activación) y comprueba que en todos el gradient check falle. Si el chequeo no los detectara, que pase en verde no probaría nada.

  Con la semilla fija, el peor error relativo de las 6 combinaciones del plan está entre 3e-10 y 6e-9: el margen contra la tolerancia es amplio.
- `test_optimizers.py`: un paso de `GD` calculado a mano, que actualice los pesos de la red in place, y que la pérdida baje en cada paso en `y = x`.
- `test_training.py`: el criterio de aceptación de la Etapa 4. Con `y = x`, 50 muestras y `GD(lr=0.1)` a lote completo, el MSE baja en todas las épocas y termina por debajo de 1e-4 (lo alcanza en la época 49 de 300). También verifica:
  - que el modo estocástico y el mini-batch converjan;
  - que `fit` a lote completo dé exactamente lo mismo que el paso manual de la Etapa 3;
  - la cantidad de pasos por época, que la misma semilla dé el mismo entrenamiento, y las columnas del historial y del CSV;
  - los callbacks: el orden de las llamadas, el corte, y que `EarlyStopping` restaure la mejor época in place, respete `mode` y `min_delta`, y corte si el entrenamiento diverge (NaN).
- `test_validation.py`: **el ejercicio previo de validación** (Etapa 5). Los casos que dependen de la inicialización se corren con 20 semillas, porque una sola corrida no alcanza para concluir.

  | Caso | Modelo | Criterio | Resultado con las 20 semillas |
  | --- | --- | --- | --- |
  | AND | `[2, 1]` escalón, `fit_perceptron`, η = 0.1 | error 0 en menos de 100 épocas | 20/20, en 1 a 4 épocas (lote completo y online) |
  | `y = x` | `[1, 1]` identidad, MSE, GD η = 0.1, 50 muestras | MSE < 1e-4 | 20/20 |
  | `y = tanh(x)` | `[1, 1]` tanh, MSE, GD η = 0.5, 50 muestras | MSE < 1e-3 | 20/20 |
  | XOR | `[2, 3, 2, 1]` tanh, MSE, GD η = 0.1, 2000 épocas | las 4 muestras bien clasificadas | 20/20 |
  | XOR | `[2, 2, 1]` tanh, MSE, GD η = 0.1, 2000 épocas | las 4 muestras bien clasificadas | 17/20; el test exige al menos 15 (ver decisión 7) |

  También verifica:
  - que el perceptrón escalón **no** pueda con XOR: nunca baja del 25% de error, porque XOR no es linealmente separable;
  - un paso de la regla del perceptrón calculado a mano;
  - que el motor reproduzca exactamente los números de `docs/verificacion_manual.md`.

---

## Decisiones de diseño a validar con el equipo

Estas decisiones no estaban fijadas en `CLAUDE.md` ni en el plan. Si alguien no está de acuerdo, conviene cambiarlas **antes de la Etapa 4**: el gradient check ya está en verde con ellas, y de la 1 depende también la escala del learning rate.

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

**Cómo lo usa `Network.backward`:** cuando la capa de salida es `Softmax`, calcula el delta de salida con `loss.softmax_delta(...)` y se lo pasa a `Dense.backward_z`, **sin** multiplicar por la derivada de la activación. En cualquier otro caso usa `Dense.backward(loss.grad(...))`, que sí multiplica por `activation.backward(z)`. Además de ser más barato, el atajo es más estable porque no divide por ŷ.

Softmax con otra pérdida (por ejemplo, MSE) lanza `ValueError` en el backward.

### 3. Variantes de inicialización

- **Xavier** es la versión **uniforme** (Glorot uniforme). Es la recomendada para capas `tanh` y `sigmoid`.
- **He** es la versión **normal**. Es la recomendada para capas `relu`.
- **Uniform** usa por default `U(−0.5, 0.5)`, la inicialización clásica del perceptrón simple.
- Los **bias arrancan en cero**; el inicializador solo genera `W`.

### 4. Cómo se construye una `Network`

- **`rng` es obligatorio y se pasa por nombre.** No hay default: si faltara, la red usaría una semilla implícita y la corrida no sería reproducible. Las capas consumen el mismo `rng` en orden, así que dos capas del mismo tamaño no arrancan con los mismos pesos.
- **Todos los argumentos, salvo `layer_sizes`, se pasan por nombre** (`hidden_activation=...`). Así no hay que recordar el orden y es más difícil confundir la activación oculta con la de salida.
- **Activaciones e inicializador aceptan un nombre o un objeto.** Con el nombre (`"tanh"`) se usan los parámetros por defecto; para parámetros propios se pasa el objeto (`Tanh(beta=2.0)`, `Uniform(-1, 1)`). El runner de la Etapa 6 va a traducir la config JSON a objetos.
- **Defaults:** capas ocultas `tanh`, salida `identity`, inicialización `xavier`.
- **Una única activación para todas las capas ocultas.** Alcanza para los tres ejercicios; si hiciera falta mezclar activaciones, se agrega sin romper esta interfaz.

### 5. Backward y optimizador

- **La pérdida se pasa al backward** (`net.backward(y, y_pred, loss)`): la red no guarda una pérdida propia. Así, en la Etapa 4, `fit` recibe la pérdida como parámetro, como dice el plan, y se puede evaluar una red con una pérdida distinta de la usada para entrenar.
- **`params` y `grads` son referencias, no copias.** El optimizador tiene que actualizar in place (`p -= lr * g`), nunca reasignar (`p = p - lr * g`): una reasignación crea un array nuevo y la capa sigue usando el viejo, así que la red no aprendería. Un test lo verifica para `GD`, y Momentum y Adam tienen que respetar lo mismo.
- **`GD` entra en esta etapa** porque el `CLAUDE.md` (sección 9, paso 3) lo pone junto al backward. En el plan aparece recién en la Etapa 4, pero es una línea y permite probar un paso de entrenamiento completo.
- **Hacer backward antes de un forward lanza `RuntimeError`**, en vez de calcular gradientes con datos viejos o inexistentes.

### 6. Entrenamiento (`fit`, historial y callbacks)

- **La pérdida de cada época se mide sobre el conjunto completo**, con los pesos del final de la época. No es el promedio de las pérdidas de los lotes, que mezcla pesos distintos a lo largo de la época y no es comparable entre `batch_size` distintos. Cuesta un forward extra por época.
- **`fit` recibe `metric`** además de lo que dice el plan. Hace falta para las columnas `train_metric` y `val_metric` del `history.csv`. Es una función `f(y_true, y_pred) -> float`; las funciones concretas (accuracy, F1, etc.) llegan con `metrics.py` en la Etapa 6.
- **La red guarda su `rng`** y lo usa para barajar los lotes, así que la semilla de la config determina tanto los pesos iniciales como el orden de las muestras. `fit(..., rng=...)` permite usar otro generador.
- **`PrintProgress` exige `write`.** El `CLAUDE.md` prohíbe `print` en `core/`, y el plan pide un callback que imprima. La salida es que el callback no elija el destino: quien llama a `fit` pasa `write=print` (o un logger, o `list.append` en los tests).
- **`EarlyStopping` restaura los mejores pesos por default** (`restore_best=True`), también cuando las épocas se agotan sin cortar. Así, lo que se evalúa y se guarda es la red de la mejor época, que es la que se reporta en `metrics.json`. Monitorear una clave que no está en los logs (por ejemplo, `val_loss` sin validación) lanza un error que explica qué falta.
- **Archivos nuevos:** `history.py` y `callbacks.py` no estaban en la estructura del `CLAUDE.md`. Los separé de `network.py` para que no creciera de más; si se prefiere, pueden vivir ahí.

### 7. Ejercicio de validación

- **La regla del perceptrón se aplica por lotes**, `ΔW = η Xᵀ (y − ŷ)`, porque el `CLAUDE.md` prohíbe recorrer las muestras de a una en Python. Con `batch_size=1` se obtiene la regla clásica, muestra a muestra. Se implementa pasándole `delta = ŷ − y` a `Dense.backward_z` y dando un paso de `GD`, así que no hace falta tocar el backprop ni duplicar la cuenta. A diferencia de las pérdidas, esta regla **no** divide por `n`: es la regla del enunciado tal cual.
- **XOR con `[2, 2, 1]` no siempre converge, y es esperable.** En 3 de las 20 semillas cae en el mínimo local clásico de XOR: dos muestras bien clasificadas, las otras dos con salida ≈ 0, pérdida ≈ 0.5 y gradiente ≈ 0. Con 20000 épocas sigue igual. No es un bug: el gradient check pasa, y es un comportamiento conocido de esta arquitectura, que tiene la capacidad justa para XOR. `[2, 3, 2, 1]` no queda atrapada con ninguna semilla. El test no elige a mano una semilla que funcione, porque eso escondería el fenómeno: exige una tasa de éxito y documenta el mínimo local con un test propio (`test_xor_2_2_1_minimo_local_conocido`). Es material útil para el informe.
- **Verificación manual:** `docs/verificacion_manual.md` tiene la iteración completa con cada fórmula y cada número, y un test comprueba que el motor da exactamente eso. Los números se calcularon con fórmulas escalares escritas una por una, sin usar el motor; **falta que alguien del equipo la rehaga en papel**, como pide el plan, para la defensa oral.
- **Pendiente opcional:** el plan sugiere graficar la frontera de decisión de AND época a época. Queda para cuando exista `analysis/plots.py`: un script de análisis no entrena, así que antes hace falta el runner de la Etapa 6.

### 8. Detalles de robustez

- **Formas estrictas en las pérdidas.** Si `y_true` e `y_pred` tienen formas distintas, se lanza `ValueError`. Sin este chequeo, `(n,)` contra `(n, 1)` se expande a `(n, n)` y la pérdida da un número creíble pero incorrecto. Los targets siempre se pasan como `(n, 1)`, nunca como `(n,)`.
- **Recorte de probabilidades.** Las entropías cruzadas recortan ŷ a `[1e-12, 1 − 1e-12]` antes del log, así la pérdida no se vuelve infinita cuando la salida satura.
