# F04 · Optimizadores, regularización y extensiones de `fit`

| | |
|---|---|
| **Objetivo** | Todo lo que el motor necesita para los ejercicios y que hoy falta: Momentum/RMSProp/Adam, η adaptativo, L2, augmentation, varias métricas por época, registro de η y detección de divergencia |
| **Requerimientos** | RF-06, RF-07 · `04-matematica.md` §2.2, §4, §5 |
| **Depende de** | Motor existente (✅) |
| **Estimación** | 1 día con el agente |
| **Regla** | Extender sin romper: firmas existentes intactas, parámetros nuevos opcionales al final. Si un test existente tiene que cambiar, el PR lo justifica y lo revisa otra persona del equipo |

## T0 · Logística con 2β (solo si el equipo aprueba alinear la logística con la cátedra)
- `Sigmoid.forward`: usar `bz = 2 * self.beta * z`; `Sigmoid.backward`: `2 * self.beta * s * (1 - s)`.
- Test nuevo: `Sigmoid(beta=b)(z) ≈ 0.5 * (1 + tanh(b*z))` para varios b.
- Actualizar la tabla de activaciones del `README.md`.
- Verificado al escribir esta spec: con ese cambio la suite actual (154 tests) sigue en verde, porque los tests de `Sigmoid` comparan contra la derivada numérica.

## T1 · Optimizadores (`core/optimizers.py`)
Fórmulas: `04-matematica.md` §4. Todos:
- implementan `step(params, grads)` **in place** (como `GD`), con estado indexado por posición y creado en el primer `step`;
- validan `lr > 0` y parámetros en rango (`0 ≤ alpha < 1`, `0 ≤ beta1, beta2, gamma < 1`, `eps > 0`);
- exponen `lr` como atributo mutable, `state_dict() -> dict` y `load_state_dict(state)`;
- se registran en `OPTIMIZERS`: `"momentum"`, `"rmsprop"`, `"adam"`, `"adagrad"` (opcional). **No** registrar `"sgd"`: un test existente usa ese nombre como desconocido.
- defaults (el test `test_registro_construye_cada_optimizador` los construye sin argumentos): Momentum `lr=0.01, alpha=0.9`; RMSProp `lr=0.001, gamma=0.9, eps=1e-8`; Adam `lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8`; AdaGrad `lr=0.01, eps=1e-8`.
- `GD` también gana `state_dict`/`load_state_dict` (triviales) para que F07 trate a todos igual.

## T2 · η adaptativo (`core/callbacks.py`)
`AdaptiveEta(optimizer, a, b, k, k_prime, monitor="train_loss", min_lr=1e-8)` según §4.1: cuenta épocas seguidas en que `monitor` baja / sube; tras `k` bajas suma `a`, tras `k_prime` subas multiplica por `(1 - b)`; reinicia contadores tras cada cambio; nunca baja de `min_lr`. Recibe el optimizador en el constructor (así no hay que cambiar la firma de los callbacks).

## T3 · Extensiones de `Network.fit` y `History`
Parámetros nuevos, opcionales, **al final** de `fit`:

| Parámetro | Comportamiento |
|---|---|
| `metrics: Mapping[str, Metric] \| None = None` | Por cada nombre, loguea `train_<nombre>` y, si hay validación, `val_<nombre>`. Convive con el `metric` existente (que sigue logueando `train_metric`/`val_metric`) |
| `l2: float = 0.0` | Después de `backward`, suma `l2 * W` a `grad_W` de cada capa (nunca a `grad_b`). Agregar `Network.l2_penalty(l2) -> float` = `l2/2 · Σ‖W‖²` para que el gradient check pueda verificarlo |
| `augment: Augmentation \| None = None` | Se aplica **solo** a los lotes de entrenamiento (`X_b = augment(X_b, aug_rng)`), nunca en `evaluate`/`predict`. `aug_rng` = generador hijo creado al inicio de `fit` a partir del `rng` de la red, para que activar augmentation no cambie el orden de los lotes |

Además:
- **Columna `lr`:** si el optimizador tiene atributo `lr`, se loguea el valor usado en la época. Se agrega `"lr"` a `COLUMNS` justo antes de `"elapsed_s"`.
- **Columnas dinámicas en `History`:** `append` acepta las claves de `COLUMNS` y cualquier clave `train_<x>` / `val_<x>`; cualquier otra sigue siendo error (el test `test_history_clave_desconocida_falla` con `"accuracy"` sigue pasando). `column(name)` acepta las dinámicas. `to_csv` escribe `COLUMNS` + las dinámicas en orden alfabético.
- **Divergencia:** si `train_loss` no es finito, se registra la época, `history.status = "diverged"` (default `"ok"`) y se corta el entrenamiento (sin esperar paciencia). `on_train_end` de los callbacks se llama igual.
- Tests existentes que cambian (justificarlo en el PR): `test_logs_sin_validacion_ni_metrica` pasa a esperar también `"lr"`.
- `fit_perceptron` no cambia.

## T4 · Augmentation (`core/augmentation.py`)
- `Augmentation` (Protocol): `__call__(X, rng) -> X_nuevo`, **nunca** modifica `X` in place.
- `GaussianNoise(sigma, clip=None)`: `X + N(0, sigma²)`, opcionalmente recortado a `clip=(lo, hi)`.
- `RandomShift(max_px, image_shape)`: reinterpreta cada fila como imagen `image_shape` (p. ej. `(8, 8)`), la traslada un entero aleatorio en `[-max_px, max_px]` en x e y, rellena con el mínimo de la fila (fondo) y vuelve a aplanar. Vectorizado sobre el lote (se permite un loop sobre los ≤ 25 desplazamientos posibles, no sobre muestras).
- `Compose([aug1, aug2])` y `get_augmentation(config_dict)` para el runner.

## Tests y criterios de aceptación
Archivos: `tests/test_optimizers.py` (ampliar), `tests/test_training.py` (ampliar), `tests/test_gradients.py` (caso L2), `tests/test_augmentation.py` (nuevo).
- [x] **Paso a mano** por optimizador con escalares (p. ej. `g = 0.5`, `η = 0.1`): Momentum 2 pasos (verifica el término `α·Δθ`); RMSProp 1 paso; Adam en `t = 1` da paso ≈ `η·sign(g)` (corrección de sesgo); AdaGrad 1 paso.
- [x] Cada optimizador minimiza `½ θᵀ diag(1, 10) θ` desde `(1, 1)` hasta `‖θ‖ < 1e-3` en ≤ 2000 pasos.
- [x] Cada optimizador actualiza **in place** los pesos de una `Network` (mismo objeto, como el test de GD).
- [x] `state_dict` → nuevo optimizador → `load_state_dict` → la trayectoria sigue idéntica a no haber cortado.
- [x] `AdaptiveEta`: con una secuencia sintética de pérdidas (vía un callback falso como `FakeLosses`) sube η en `a` tras `k` bajas, lo baja a `(1-b)η` tras `k'` subas, reinicia contadores, respeta `min_lr`; la columna `lr` del historial refleja los cambios.
- [x] Gradient check de `[3, 4, 2]` tanh + MSE **con L2**: el gradiente de `loss.value + net.l2_penalty(l2)` coincide con el analítico; los bias no reciben término L2.
- [x] Con `l2` grande, la norma de los pesos entrenados es menor que con `l2 = 0` (misma semilla).
- [x] `metrics={"mae": f}` produce `train_mae` y `val_mae` en `history.records` y en el CSV.
- [x] Divergencia: `y = x` con `GD(lr=1e6)` termina con `history.status == "diverged"` antes de agotar las épocas y sin excepción.
- [x] Augmentation: no altera `X` original; `sigma=0` es identidad; reproducible con la misma semilla; `RandomShift` de una imagen con un solo píxel encendido lo mueve a una posición válida; `evaluate` no aplica augmentation (con `sigma` enorme, `evaluate` da lo mismo que sin augmentation).
- [x] XOR `[2,2,1]` con GD, Momentum y Adam (20 semillas, xavier): registrar en el log del test la mediana de épocas hasta resolver de cada uno (sin assertar un orden) → dato para la presentación.
- [x] Toda la suite anterior sigue en verde (salvo el test justificado arriba) y `ruff` limpio.

## Entregables
Rama `f04-optimizers` lista para PR, con el `README.md` actualizado (tabla de optimizadores, callbacks y parámetros nuevos de `fit`). El agente propone los mensajes de commit (`feat(F04): ...`) y no commitea sin permiso (`CLAUDE.md` §0).