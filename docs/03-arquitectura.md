# 03 · Arquitectura

Describe **lo que ya existe** en el repo (motor existente, Etapas 0–5 de `PLAN_MOTOR.md`) y **lo que agregan** las fases F04–F07. Regla: extender sin romper (Constitución C2).

## 1. Estructura

Ver `CLAUDE.md` §4 (árbol completo con el estado de cada módulo). Archivos que crea o modifica cada fase:

| Fase | Archivos |
|---|---|
| F04 | `core/optimizers.py`, `core/callbacks.py`, `core/network.py`, `core/history.py`, `core/augmentation.py`, `core/activations.py` |
| F05 | `data/loaders.py`, `data/preprocess.py`, `data/splits.py`, `data/synthetic.py` |
| F06 | `core/metrics.py`, `core/thresholds.py` |
| F07 | `core/serialization.py`, `experiments/config.py`, `experiments/runner.py`, `analysis/common.py`, `analysis/aggregate.py`, `experiments/configs/validacion/` |
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
save_checkpoint(path, network, optimizer=None, extra: dict | None = None)   # npz + json embebido
load_checkpoint(path) -> (network, optimizer | None, extra)                 # permite seguir entrenando

# experiments/config.py
load_config(path) -> dict (defaults resueltos + validación) · expand(config) -> list[dict] (sweep × seeds [× folds])
config_hash(config_sin_seed) -> str (8 hex)

# experiments/runner.py (CLI)
python -m experiments.runner <config.json> [--smoke] [--resume <run_dir>] [--final-eval] [--force] [--workers N]

# analysis/common.py · analysis/aggregate.py
load_runs(results_dir) -> DataFrame · load_histories(results_dir) -> DataFrame (largo) · aggregate(df, by)
```

## 4. Contrato de configuración (JSON)

Base en `CLAUDE.md` §6. Claves completas y defaults (los define `experiments/config.py`):

| Clave | Default | Notas |
|---|---|---|
| `run_name` | obligatorio | carpeta en `results/` |
| `seed` / `seeds` | `[0, 1, 2]` | `seed` suelto = `[seed]` |
| `dataset.path` | — | o `dataset.synthetic: {"name": "and"\|"xor"\|"line", ...}` para validación |
| `dataset.target`, `features`, `drop`, `categorical`, `na_policy` | —, todas, `[]`, `[]`, `"error"` | nombres exactos en `docs/datos/*.md` |
| `dataset.normalize` | `"minmax"` | `none`, `minmax`, `zscore`, `unit_length`; `feature_range` |
| `dataset.target_encoding` | `"none"` | `onehot`, `pm1_onehot` (−1/+1 para tanh), `scale_to_output` |
| `dataset.split` | `{"kind": "holdout", "ratio": 0.8, "stratified": true}` | `none`, `holdout`, `kfold` (`k`) |
| `dataset.holdout_test` | `null` | Ej1: `{"ratio": 0.2, "seed": 1234, "indices_path": "results/ej1_split/test_idx.npy"}`; desarrollo **excluye** esos índices |
| `dataset.test_path` | `null` | solo con `--final-eval` (+ `selected_from`) |
| `model.layers`, `hidden_activation`, `output_activation`, `beta`, `initializer` (+ `init_params`) | — , `tanh`, `identity`, `1.0`, `xavier` | `layers[0]` se puede poner `"auto"` = nº de features |
| `training.loss` | `"mse"` | |
| `training.optimizer` | `{"kind": "gd", "lr": 0.01}` | + parámetros propios de cada optimizador |
| `training.batch_size` | `null` (lote completo) | `1` = online |
| `training.epochs`, `l2`, `augmentation` | —, `0.0`, `null` | |
| `training.early_stopping`, `adaptive_eta` | `null` | |
| `metrics` | `[]` | nombres de `get_metric`; `task` se infiere del dataset |
| `logging.every`, `save_model`, `save_predictions` | `10`, `true`, `false` | |
| `sweep` | `{}` | producto cartesiano, claves con notación punto |
| `selected_from` | `null` | solo `--final-eval`: ruta del run elegido |

## 5. Contrato de resultados

`CLAUDE.md` §7: `results/<run_name>/<hash8>_s<seed>[_f<fold>]/` con `config.json`, `history.csv`, `metrics.json`, `predictions.npz`, `model.npz`. Además:
- `results/<run_name>/summary.csv` (generado por `analysis.aggregate`): una fila por `hash8` con media, desvío, min, max y `n` de cada métrica final + las claves del sweep.
- `results/<run_name>/final_eval.json` (solo `--final-eval`): métricas en test, matriz de confusión, `selected_from`.
- Relanzar un barrido **saltea** runs completos (tienen `metrics.json`); `--force` los rehace.