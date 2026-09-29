# 03 · Arquitectura

Describe **lo que ya existe** en el repo (motor existente, Etapas 0–5 de `PLAN_MOTOR.md`) y **lo que agregan** las fases F04–F07. Regla: extender sin romper (Constitución C2).

## 1. Estructura

Ver `CLAUDE.md` §4 (árbol completo con el estado de cada módulo). Archivos que crea o modifica cada fase:

| Fase | Archivos |
|---|---|
| F04 | `core/optimizers.py`, `core/callbacks.py`, `core/network.py`, `core/history.py`, `core/augmentation.py`, `core/activations.py` |
| F05 | `data/loaders.py`, `data/preprocess.py`, `data/splits.py`, `data/synthetic.py` |
| F06 | `core/metrics.py`, `core/thresholds.py` |
| F07 | `core/serialization.py`, `experiments/config.py`, `experiments/runner.py`, `analysis/common.py`, `analysis/aggregate.py`, `experiments/configs/validacion/`; extensiones: `core/callbacks.py` (`state_dict`), `core/network.py` (atributo `initializer`), `data/splits.py` (`prepare_fold` sin validación) |
| F09–F11 | `notebooks/`, `experiments/configs/ej1/`, `analysis/ej1_*.py`, `docs/datos/fraud_dataset.md`, `docs/resultados/ej1.md` |
| F12–F13 | `notebooks/`, `experiments/configs/ej2/`, `experiments/configs/ej3/`, `analysis/ej2*.py`, `analysis/ej3.py`, `docs/datos/digits.md`, `docs/resultados/ej2.md`, `docs/resultados/ej3.md` |
| F14 | `analysis/plots.py`, `analysis/figures.py`, `figures/` |

F04, F05 y F06 tocan archivos distintos, así que se pueden implementar en paralelo sin conflictos de merge.

## 2. Interfaces existentes (no cambiar firmas)

Convención de formas: `X (n_muestras, n_features)`, `y (n_muestras, n_salidas)`; una sola salida es `(n, 1)`.

```python
# core/activations.py
class Activation(Protocol):
    def forward(self, z: np.ndarray) -> np.ndarray: ...
    def backward(self, z: np.ndarray) -> np.ndarray: ...   # derivada evaluada en z
get_activation(name, **kwargs)   # "step", "identity", "tanh"(beta), "sigmoid"(beta), "relu", "softmax"

# core/losses.py
class Loss(Protocol):
    def value(self, y_true, y_pred) -> float: ...           # promediada (ver 04-matematica §2.4)
    def grad(self, y_true, y_pred) -> np.ndarray: ...       # dL/dŷ, ya dividido por n (y m en MSE/BCE)
get_loss(name)   # "mse", "binary_crossentropy", "categorical_crossentropy" (+ softmax_delta)

# core/initializers.py
Initializer.__call__(n_in, n_out, rng) -> W (n_in, n_out)   # "uniform"(low, high), "xavier", "he"; bias en 0

# core/layers.py
Dense(n_in, n_out, activation, initializer, rng)
    .W (n_in, n_out) · .b (1, n_out) · .forward(x) · .backward(grad_a) · .backward_z(delta)
    .params -> [W, b] · .grads -> [grad_W, grad_b] · .n_params

# core/network.py
Network(layer_sizes, *, hidden_activation="tanh", output_activation="identity",
        initializer="xavier", rng)
    .predict(X) · .evaluate(X, y, loss, metric=None) -> {"loss", "metric"?}
    .fit(X, y, optimizer, loss, epochs, batch_size=None, validation=None,
         callbacks=(), metric=None, rng=None) -> History
    .backward(y_true, y_pred, loss) · .params · .grads · .n_params · .layers · .rng

# core/perceptron.py
fit_perceptron(network, X, y, lr, epochs, batch_size=None, callbacks=(), rng=None) -> History

# core/optimizers.py
class Optimizer(Protocol):
    def step(self, params: Sequence[np.ndarray], grads: Sequence[np.ndarray]) -> None: ...  # IN PLACE
GD(lr)

# core/callbacks.py
class Callback: on_train_begin(network) · on_epoch_end(epoch, logs, network) -> bool (True corta) · on_train_end(network)
PrintProgress(every, write) · EarlyStopping(patience, monitor="val_loss", mode="min", min_delta=0.0, restore_best=True)

# core/history.py
History: .records · .append(logs) · .column(name) · .to_csv(path) · COLUMNS
```

## 3. Extensiones que agregan las fases

### F04 — sin romper lo anterior
```python
# core/optimizers.py — mismas reglas que GD: in place, estado propio, registro por nombre
Momentum(lr, alpha=0.9) · RMSProp(lr, gamma=0.9, eps=1e-8) · Adam(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)
AdaGrad(lr, eps=1e-8)  # opcional
optimizer.lr                      # atributo mutable (lo usa AdaptiveEta)
optimizer.state_dict() -> dict    # arrays + escalares, serializable con np.savez
optimizer.load_state_dict(state)

# core/callbacks.py
AdaptiveEta(optimizer, a, b, k, k_prime, monitor="train_loss")

# core/augmentation.py
class Augmentation(Protocol):
    def __call__(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...  # nunca in place
GaussianNoise(sigma, clip=None) · RandomShift(max_px, image_shape) · Compose([...])

# core/network.py — parámetros NUEVOS, opcionales, al final de fit
fit(..., metrics: Mapping[str, Metric] | None = None,   # además del `metric` existente
         l2: float = 0.0,                                # suma l2 * W a grad_W (no a b) antes del step
         augment: Augmentation | None = None)            # se aplica solo a los lotes de train

# core/history.py
# columnas dinámicas: COLUMNS base + "lr" + f"train_{m}" / f"val_{m}" por cada métrica de `metrics`
# history.status: "ok" | "diverged"   (fit corta si train_loss no es finito)
```

### F05
```python
# data/loaders.py
@dataclass
class Dataset: X (p, n) float64 · y (p,) o (p, k) · feature_names · target_name · meta (path, sha256, filas descartadas)
load_csv(path, target, features=None, drop=(), categorical=(), na_policy="error") -> Dataset

# data/preprocess.py — fit(X) -> self · transform · inverse_transform · state_dict
MinMaxScaler(feature_range=(0, 1)) · StandardScaler() · UnitLengthScaler() · IdentityScaler()
TargetScaler(out_range)                 # lleva y a la imagen de la activación de salida y vuelve
one_hot(y, n_classes, neg=0.0, pos=1.0) · get_scaler(name, **kw)

# data/splits.py — devuelven ÍNDICES
holdout(n, ratio, rng) · stratified_holdout(y, ratio, rng) · kfold(n, k, rng) · stratified_kfold(y, k, rng)
stratify_labels(y, n_bins=10)            # para targets continuos
prepare_fold(dataset, idx_train, idx_val, normalize="minmax", target_encoding="none",
             n_classes=None, out_range=None) -> (X_tr, y_tr, X_val, y_val, fitted)

# data/synthetic.py
and_dataset() · xor_dataset() · line_samples(f, n, lo, hi, rng)
```

### F06
```python
# core/metrics.py — nivel etiquetas
confusion_matrix(y_true, y_pred, n_classes=None)   # filas = real, columnas = predicho
accuracy · precision · recall · f1 · fbeta · tpr · fpr · per_class_report · macro · mse · rmse · mae
classification_summary(y_true_labels, y_pred_labels, n_classes) -> dict (JSON)
# nivel salidas de la red (compatible con Metric de fit: f(y_true, y_pred) con arrays (n, n_out))
get_metric(name, *, task: "binary" | "multiclass" | "regression", threshold=0.5) -> Metric
METRIC_NAMES = ("accuracy", "macro_f1", "precision", "recall", "f1", "mse", "mae", ...)

# core/thresholds.py
threshold_sweep(y_true, y_score, thresholds, cost_fn=1, cost_fp=1) -> pd.DataFrame
roc_curve · pr_curve · auc_trapezoid · average_precision · select_threshold(df, criterion, min_recall=None)
```

### F07
```python
# core/serialization.py
save_checkpoint(path, network, optimizer=None, extra: dict | None = None)   # npz sin pickle + JSON embebido
load_checkpoint(path) -> (network, optimizer | None, extra)                 # permite seguir entrenando
Checkpoint(path, every, optimizer=None, extra=None)   # callback: save_checkpoint cada `every` épocas
# extra admite dicts, listas, escalares y ndarrays (van como arrays aparte); el estado
# del rng de la red (rng.bit_generator.state) se guarda siempre.

# core/callbacks.py — métodos nuevos para reanudar (se aplican DESPUÉS de on_train_begin)
EarlyStopping.state_dict() / .load_state_dict(state)   # best, best_epoch, stopped_epoch, wait, best_params
AdaptiveEta.state_dict() / .load_state_dict(state)     # valor anterior y contadores (el η vive en el optimizador)

# core/network.py — atributo nuevo
Network.initializer   # el inicializador usado, para serializar la red

# experiments/config.py
load_config(path, *, final_eval=False) -> dict   # defaults + validación (ConfigError nombra la clave)
resolve_config(dict, *, final_eval=False) · load_run_config(run_dir/config.json)
expand(config) -> list[dict]      # sweep × seeds [× folds]; cada corrida tiene seed y fold
config_hash(config) -> str        # sha1, 8 hex, sin run_name, seed(s), fold, sweep ni logging
run_id(config) -> "<hash8>_s<seed>[_f<fold>]"
build(config, rng, *, n_features=None, n_outputs=None, task=None, threshold=None)
    -> (network, optimizer | None, loss, callbacks, metrics, augment)

# experiments/runner.py (CLI)
python -m experiments.runner <config.json> [--smoke] [--force] [--only CLAVE=VALOR]... [--workers N]
                                           [--results-dir DIR]
python -m experiments.runner --resume <run_dir>
python -m experiments.runner <config.json> --final-eval [--force]
run_experiment(path, ...) · train_run(config, run_dir, ...) · resume_run(run_dir) · final_eval(path, ...)

# analysis/common.py · analysis/aggregate.py
load_runs(results_dir) -> DataFrame · load_histories(results_dir) -> DataFrame (largo)
aggregate(df, by, metrics=None) -> DataFrame (n, media, desvío ddof=1, min, max)
apply_style() · save_figure(fig, path) -> [png 200 dpi, pdf] · PALETTE
python -m analysis.aggregate results/<run_name>   # → summary.csv
```

Detalles del runner (F07):
- **Datos:** se excluyen los índices de `holdout_test`; el resto se particiona con un generador propio, `default_rng([split.seed o seed, 1])`, así que cambiar el modelo no cambia la partición. El modelo usa `default_rng(seed)`, que decide los pesos iniciales y el orden de los lotes.
- **Tarea** (`dataset.task: "auto"`): multiclass con one-hot, binary si el target tiene 2 valores y regression en otro caso. El umbral binario es el punto medio de los dos valores del target codificado: 0 con ±1 (tanh) y 0.5 con 0/1 (sigmoide).
- **Reanudar:** el checkpoint guarda pesos, optimizador, rng de la red, estado de los callbacks, época, tiempo e historial. Reanudar sin augmentation da exactamente lo mismo que no cortar. Con augmentation no: `fit` crea en cada llamada un generador hijo (`rng.spawn`) cuyo estado no se guarda.
- **Split `none`:** sin validación (Ej1, R-01). `predictions.npz` guarda entonces las predicciones de train (`subset = "train"`).

## 4. Contrato de configuración (JSON)

Base en `CLAUDE.md` §6. Claves completas y defaults (los define `experiments/config.py`):

| Clave | Default | Notas |
|---|---|---|
| `run_name` | obligatorio | carpeta en `results/`; letras, números, `_`, `.`, `-` |
| `seed` / `seeds` | `[0, 1, 2]` | `seed` suelto = `[seed]`; no las dos |
| `trainer` | `"backprop"` | `"perceptron"`: llama a `fit_perceptron` con `training.optimizer.lr` (solo `{"kind": "gd", "lr": η}`), para `layers [n, 1]` con salida `step` |
| `dataset.name` | `null` | etiqueta descriptiva (no cambia nada) |
| `dataset.path` | — | o `dataset.synthetic` (uno solo) |
| `dataset.synthetic` | `null` | `{"name": "and"}`, `{"name": "xor"}` o `{"name": "line", "function": "identity"\|"tanh", "n": 50, "low": -1.0, "high": 1.0, "seed": 0}` |
| `dataset.format` | `"tabular"` | `"tabular"` → `load_csv`; `"digits"` → `load_digits_csv` (784 features, target `label`) |
| `dataset.target`, `features`, `drop`, `categorical`, `na_policy` | —, todas, `[]`, `[]`, `"error"` | solo `tabular`; nombres exactos en `docs/datos/*.md` |
| `dataset.task` | `"auto"` | `binary`, `multiclass`, `regression`; `auto`: multiclass con one-hot, binary si el target tiene 2 valores, si no regression |
| `dataset.n_classes` | `null` | solo con one-hot; `null` = máx. etiqueta + 1 (en `digits`, 10) |
| `dataset.normalize` | `"minmax"` | `none`, `minmax`, `zscore`, `unit_length` |
| `dataset.feature_range` | `null` (= `[0, 1]`) | solo con `minmax` |
| `dataset.target_encoding` | `"none"` | `onehot`, `pm1_onehot` (−1/+1 para tanh), `scale_to_output` (salida tanh o sigmoid) |
| `dataset.target_in_range` | `null` | solo con `scale_to_output`: rango fijo de ζ (p. ej. `[0, 1]` para probabilidades) |
| `dataset.split` | `{"kind": "holdout", "ratio": 0.8, "stratified": true, "seed": null}` | `none` (sin validación), `holdout` (`ratio` = fracción de train), `kfold` (`k`: 5, `stratified`, `seed`). `seed: null` = semilla de la corrida; un entero fija la partición entre semillas. Target continuo: se estratifica por cuantiles |
| `dataset.holdout_test` | `null` | `{"ratio": 0.2, "seed": 0, "stratified": true, "indices_path": null}`; desarrollo **excluye** esos índices. Con `indices_path` se generan una vez y se reusan (Ej1: `"results/ej1_split/test_idx.npy"`). Solo se evalúa con `--final-eval` |
| `dataset.test_path` | `null` | solo con `--final-eval` (+ `selected_from`); mismo `format` que `path` |
| `model.layers`, `hidden_activation`, `output_activation`, `beta`, `initializer` (+ `init_params`) | — , `tanh`, `identity`, `1.0`, `xavier`, `{}` | `layers[0]` se puede poner `"auto"` = nº de features; `beta` va a toda capa tanh/sigmoid; `init_params` p. ej. `{"low": -0.1, "high": 0.1}` |
| `training.loss` | `"mse"` | `softmax` exige `categorical_crossentropy` (y one-hot); `binary_crossentropy` exige `sigmoid` |
| `training.optimizer` | `{"kind": "gd", "lr": 0.01}` | + parámetros propios de cada optimizador; sin `lr`, el default de la clase (`0.001` en `rmsprop` y `adam`) |
| `training.batch_size` | `null` (lote completo) | `1` = online |
| `training.epochs`, `l2`, `augmentation` | —, `0.0`, `null` | `augmentation`: dict o lista de dicts de `get_augmentation`; `random_shift` en `digits` completa `image_shape: [28, 28]` |
| `training.early_stopping`, `adaptive_eta` | `null` | kwargs de `EarlyStopping` / `AdaptiveEta` (sin `optimizer`), con sus defaults; `monitor` tiene que estar en los logs y `mode` coincidir (`val_accuracy` → `max`) |
| `metrics` | `[]` | nombres de `get_metric`; `task` se infiere del dataset |
| `logging.every`, `save_model`, `save_predictions`, `checkpoint_every` | `10`, `true`, `false`, `50` | `checkpoint_every`: épocas entre checkpoints (`0` o `null` = ninguno) |
| `sweep` | `{}` | producto cartesiano, claves con notación punto (una sección entera también, p. ej. `training.optimizer`); no se pueden barrer `run_name`, `seed(s)` ni `logging.*` |
| `selected_from` | `null` | solo `--final-eval`: carpeta de una corrida o prefijo `results/<run_name>/<hash8>` (todas sus semillas/folds) |
| `final` | `null` | solo `--final-eval`: `{"mode": "retrain"\|"reuse", "epochs": null, "threshold": null}`. `retrain`: reentrena con todo desarrollo durante `epochs` o la mediana de `best_epoch` de `selected_from`, una vez por semilla; `reuse`: usa sus `model.npz`. `threshold` reemplaza al umbral binario |

## 5. Contrato de resultados

`CLAUDE.md` §7: `results/<run_name>/<hash8>_s<seed>[_f<fold>]/` con `config.json`, `history.csv`, `metrics.json`, `predictions.npz`, `model.npz`. Además:
- `metrics.json`: `run_id`, `hash`, `seed`, `fold`, `status` (`ok`/`diverged`), `task`, `threshold`, `n_classes`, `epochs`, `epochs_trained`, `best_epoch` (el de `EarlyStopping`; sin él, la época de menor `val_loss` o la última), `stopped_epoch`, `time_total_s`, `s_per_epoch`, `n_params`, `lr_final`, `resumed_from_epoch`, `smoke`, `git_commit` (con `-dirty` si hay cambios sin commitear en el código), versiones, `data` (sha256, filas de train/val/test excluidas) y `train`/`val`: `loss`, las métricas pedidas y `classification_summary` (clasificación) o `mse`, `rmse`, `mae` (regresión). Se escribe último: una corrida con `metrics.json` está completa.
- `predictions.npz`: `y_true` (target codificado), `y_score`, `idx` (filas del dataset), `subset` (`val`, o `train` sin validación) y, en clasificación, `labels_true`/`labels_pred`.
- `model.npz`: `save_checkpoint` con `extra = {config, epoch, preprocessing}`; `preprocessing` tiene el scaler ajustado, la codificación del target, la tarea y el umbral, así el modelo se aplica solo a datos crudos nuevos.
- `checkpoint.npz`: mientras entrena, cada `logging.checkpoint_every` épocas; se borra al terminar. `error.txt` si la corrida falló (el barrido sigue).
- `results/<run_name>/log.txt`: inicio y fin de cada lanzamiento, cada corrida `[i/N]` con su estado, duración y tiempo restante estimado, y cada lectura del test.
- `results/<run_name>/summary.csv` (generado por `analysis.aggregate`): una fila por `hash8` con las claves del sweep, `n`, `n_ok`, `n_diverged` y media, desvío, min y max de cada métrica final (`train.*`, `val.*`, `best_epoch`, `epochs_trained`, `time_total_s`, `s_per_epoch`, `n_params`).
- `results/<run_name>/final_eval.json` (solo `--final-eval`): métricas en test de cada modelo (con matriz de confusión), media y desvío entre modelos, `selected_from`, corridas elegidas, épocas, origen y sha256 del test. Si ya existe, `--final-eval` se niega (salvo `--force`): el test se evalúa una sola vez. Los modelos reentrenados quedan en `final_s<seed>/`; las predicciones en test, en `test_predictions.npz`.
- `results/_smoke/<run_name>/`: corridas de `--smoke` (siempre se rehacen).
- Relanzar un barrido **saltea** runs completos (tienen `metrics.json`); `--force` los rehace.