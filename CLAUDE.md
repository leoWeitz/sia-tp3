# CLAUDE.md — TP3 SIA: Perceptrón Simple y Multicapa

Guía para cualquier agente o persona que trabaje en este repo. Leer entero antes de escribir código.

---

## 1. Qué es este proyecto

Trabajo práctico N°3 de Sistemas de Inteligencia Artificial (ITBA, 2C 2026). Implementamos **desde cero** un motor de redes neuronales feed-forward y lo usamos para tres estudios experimentales:

- **Ej. 1** — Knowledge distillation: aproximar la probabilidad de fraude de `BigModel` con un perceptrón simple (lineal y no lineal) sobre `transactions.csv`.
- **Ej. 2** — Clasificación de dígitos 0–9 con perceptrón multicapa sobre `digits.csv` (ajuste) y `digits_test.csv` (producción).
- **Ej. 3** — Mismo problema con `more_data_digits.csv`, objetivo accuracy ≥ 98%.

Equipo de 4 personas. Entregable final: código + informe + presentación.

---

## 2. Reglas duras (no negociables)

1. **Prohibido usar librerías de ML para el modelo.** Nada de scikit-learn, PyTorch, TensorFlow o Keras para definir, entrenar o hacer backprop. Se permiten NumPy, pandas, matplotlib, y scikit-learn **solo** para splits, métricas o baselines de comparación, dejándolo explícito en el informe.
2. **Los tests no definen hiperparámetros** `digits_test.csv` y el test del Ej. 3 no participan de ninguna decisión de hiperparámetros. Toda decisión sale de un split de validación derivado del conjunto de entrenamiento.
3. **Toda corrida es reproducible.** Semilla explícita en la config, guardada junto a los resultados. Si un gráfico del informe no se puede regenerar desde su `config.json`, no va al informe.
4. **Operaciones matriciales.** Forward y backward por lotes con matrices `(n_muestras, n_features)`. Prohibido el loop por muestra en Python dentro del camino de entrenamiento.
5. **Ningún cambio al motor se mergea sin que pase `pytest` y el gradient check numérico.**
6. **`experiments/` entrena y escribe; `analysis/` solo lee.** Un script de análisis nunca entrena un modelo.

---

## 3. Stack y entorno

| Cosa | Elección |
| --- | --- |
| Lenguaje | Python 3.10+ |
| Cálculo | NumPy |
| Datos | pandas |
| Gráficos | matplotlib |
| Tests | pytest |
| Entorno | `venv` + `requirements.txt` (o `uv`) |
| Formato/lint | `ruff format` + `ruff check` |

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                        # suite completa, incluye gradient check
python -m experiments.runner experiments/configs/ej2_base.json
python -m analysis.plots --run results/ej2_base_20261005_143012
```

---

## 4. Estructura del repo

```
sia_tp3/
  core/
    activations.py      # step, identity, tanh, sigmoid, relu, softmax (+ derivadas)
    losses.py           # MSE, BCE, categorical cross-entropy (+ gradientes)
    initializers.py     # uniform, xavier, he
    layers.py           # Dense: forward, backward, acumulación de gradientes
    network.py          # Network: fit, predict, evaluate, save, load
    optimizers.py       # GD, Momentum, Adam
    metrics.py          # mse, mae, accuracy, precision, recall, f1, confusion_matrix
  data/
    loaders.py          # lectura de cada CSV del TP
    preprocess.py       # normalización de features, escalado de target, one-hot
    splits.py           # holdout, k-fold, estratificado
  experiments/
    runner.py           # config JSON -> entrenamiento -> results/<run_id>/
    configs/            # un JSON por experimento, versionado en git
  analysis/
    plots.py            # curvas, barridos, matriz de confusión, fronteras de decisión
    tables.py           # tablas comparativas para el informe
  tests/
    test_activations.py
    test_gradients.py   # gradient check numérico
    test_validation.py  # AND, y=x, y=tanh(x), XOR
  results/              # gitignored salvo metrics.json y config.json
  datasets/             # CSVs provistos por la cátedra (gitignored si son pesados)
  docs/                 # informe y material de la presentación
```

---

## 5. Convenciones de código

- **Idioma:** código, nombres e identificadores en inglés; docstrings, comentarios, informe y mensajes de commit en español.
- **Formas de arrays:** siempre `(n_muestras, n_features)`. Documentar la forma esperada en el docstring de toda función que reciba o devuelva arrays.
- **Bias separado:** cada capa guarda `W` de forma `(n_in, n_out)` y `b` de forma `(1, n_out)`. No agregar una columna de unos al dataset.
- **Activaciones y pérdidas como objetos** con métodos `forward(x)` y `backward(x)` (o `grad`), no como funciones sueltas: así el optimizador y la red no necesitan saber cuál están usando.
- **Sin estado global.** Nada de variables de módulo mutables; toda configuración se pasa por parámetro.
- **Sin `print` en `core/`.** El reporte de progreso se hace vía callbacks que recibe `fit`.
- **Type hints** en toda función pública.
- **Semilla:** el `Network` recibe un `numpy.random.Generator` ya construido; nunca llama a `np.random.seed` global.

---

## 6. Contrato de configuración

Todo experimento se describe con un JSON. Agregar un hiperparámetro nunca debe requerir tocar `core/`.

```json
{
  "run_name": "ej2_base",
  "seed": 42,
  "dataset": {
    "name": "digits",
    "path": "datasets/digits.csv",
    "normalize": "minmax",
    "split": { "kind": "holdout", "ratio": 0.8, "stratified": true }
  },
  "model": {
    "layers": [784, 64, 10],
    "hidden_activation": "tanh",
    "output_activation": "softmax",
    "initializer": "xavier"
  },
  "training": {
    "loss": "categorical_crossentropy",
    "optimizer": { "kind": "adam", "lr": 0.001, "beta1": 0.9, "beta2": 0.999 },
    "batch_size": 32,
    "epochs": 200,
    "early_stopping": { "patience": 20, "monitor": "val_loss" }
  },
  "logging": { "every": 10, "save_model": true }
}
```

Reglas:

- Toda clave tiene un default en `runner.py`; un JSON mínimo tiene que correr.
- Los JSON de experimentos del informe se versionan en git.
- Un campo desconocido en el JSON es un **error**, no se ignora en silencio.

---

## 7. Contrato de resultados

Cada corrida escribe un directorio `results/<run_name>_<timestamp>/`:

| Archivo | Contenido |
| --- | --- |
| `config.json` | La config exacta usada, con defaults ya resueltos |
| `history.csv` | Una fila por época: `epoch, train_loss, val_loss, train_metric, val_metric, elapsed_s` |
| `metrics.json` | Métricas finales, mejor época, tiempo total, cantidad de parámetros |
| `model.npz` | Pesos, arquitectura y estado del optimizador (si `save_model`) |

`analysis/` consume exclusivamente estos archivos. Nunca hardcodear números del informe: salen de acá.

---

## 8. Git

- Se versionan sí o sí: `experiments/configs/`, `results/**/config.json`, `results/**/metrics.json`.

---

## 9. Orden de implementación

No saltear etapas. Cada una tiene que estar verde antes de la siguiente.

1. `activations.py`, `losses.py`, `initializers.py` + sus tests unitarios.
2. `layers.py` y `network.py` con forward.
3. Backward + `optimizers.GD` + **gradient check numérico**.
4. Ejercicio previo de validación: AND, y=x, y=tanh(x), XOR `[2,2,1]` y `[2,3,2,1]`.
5. `runner.py`, configs, logging y `save`/`load`.
6. Momentum, Adam, mini-batch.
7. Ejercicio 1 → Ejercicio 2 → Ejercicio 3.
8. Opcionales, **solo** con los tres ejercicios cerrados.

---

## 10. Errores comunes

- Usar el conjunto de test para elegir hiperparámetros.
- Comparar dos configuraciones corridas con semillas distintas.
- Concluir sobre una sola corrida sin reportar desvío entre semillas.
- Olvidar escalar el target cuando la salida es `tanh` o `sigmoid`.
- No normalizar las features y concluir que "el modelo no aprende".
- Reportar accuracy sola en un problema desbalanceado.
- Gráficos sin ejes rotulados, sin unidades o sin indicar la configuración que representan.
- Descubrir un bug de backprop en la última semana.

---

## 11. Si sos un agente trabajando en este repo

- Antes de tocar `core/`, leé `tests/test_gradients.py` y `tests/test_validation.py`: definen el comportamiento esperado.
- Después de cualquier cambio en `core/`, corré `pytest` completo y pegá el resultado.
- No agregues dependencias sin actualizar `requirements.txt`.
- No modifiques archivos bajo `results/`.
- No inventes números para el informe: si hace falta un resultado, corré el experimento y citá el `run_id`.
- Si una decisión de diseño no está en este archivo, proponela en el PR en vez de asumirla.