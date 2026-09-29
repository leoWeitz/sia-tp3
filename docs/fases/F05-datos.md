# F05 · Datos: carga, preprocesamiento y particiones

| | |
|---|---|
| **Objetivo** | Utilidades de datos genéricas y seguras contra fugas de información (data leakage) |
| **Requerimientos** | RF-08, RF-09 · `04-matematica.md` §8 · Constitución C5 |
| **Depende de** | Nada del motor (solo NumPy/pandas). Conviene hacerla **después de la EDA de F09** para saber si hay categóricas, NaN o IDs |
| **Estimación** | medio día con el agente |

## Especificación

### `data/loaders.py`
```python
@dataclass
class Dataset:
    X: np.ndarray              # (p, n) float64
    y: np.ndarray              # (p,) o (p, k) — target crudo
    feature_names: list[str]
    target_name: str
    meta: dict                 # path, sha256 del archivo, filas descartadas, columnas codificadas

def load_csv(path, target: str, features: list[str] | None = None, drop: Sequence[str] = (),
             categorical: Sequence[str] = (), na_policy: Literal["error", "drop_rows"] = "error") -> Dataset
```
- Valida que existan las columnas; error claro si no.
- `categorical`: columnas que se codifican one-hot (nombres `col=valor`), solo si la EDA (F09) lo pide.
- **No** imputa ni limpia en silencio: con `na_policy="error"` informa NaN por columna; `"drop_rows"` descarta y lo registra en `meta`.
- Guarda el sha256 del archivo en `meta` (el runner lo copia a `metrics.json`).

```python
def load_digits_csv(path, cache: bool = True) -> Dataset
```
- Formato real de `digits.csv`, `digits_test.csv` y `more_digits.csv` (verificado): dos columnas, `label` (entero 0–9) e `image` (**texto** con una lista de 784 floats en [0, 1], imagen 28×28 aplanada por filas). `load_csv` no sirve para este formato.
- Parsear `image` con `json.loads` y apilar en `X (p, 784)` float64 (≈ 3 s para `digits.csv`; el loader de la cátedra usa `ast.literal_eval`, que es equivalente pero más lento). `y` = `label` como entero.
- `feature_names = ["px_0", …, "px_783"]`, `meta` con `image_shape = (28, 28)`, sha256 y cantidad de filas.
- `cache=True`: la primera vez guarda `X`, `y` en un `.npz` al lado del CSV (en `datasets/`, que está en `.gitignore`) y las siguientes lo carga si el sha256 del CSV coincide.

### `data/preprocess.py`
- Scalers con API `fit(X) -> self`, `transform(X)`, `inverse_transform(X)`, `state_dict()` / `from_state_dict()`:
  - `MinMaxScaler(feature_range=(0, 1))`, `StandardScaler()`, `UnitLengthScaler()` (por fila, sin estado), `IdentityScaler()`.
- `TargetScaler(out_range)`: lleva ζ a la imagen de la activación de salida (p. ej. `(0, 1)` sigmoide, `(-1, 1)` tanh) e invierte al predecir. Para Ej1 con probabilidades y salida sigmoide es identidad; existe para lineal/tanh.
- `one_hot(y, n_classes, neg=0.0, pos=1.0)` — `neg=-1.0` para salida tanh.
- `get_scaler(name, **kw)` con nombres `"none" | "minmax" | "zscore" | "unit_length"`.

### `data/splits.py` — devuelven **índices**, nunca copias
```python
holdout(n, ratio, rng) -> (idx_train, idx_val)                 # ratio = fracción de train
stratified_holdout(y, ratio, rng) -> (idx_train, idx_val)
kfold(n, k, rng) -> list[(idx_train, idx_val)]
stratified_kfold(y, k, rng) -> list[(idx_train, idx_val)]
stratify_labels(y, n_bins=10) -> np.ndarray                    # para targets continuos (prob. de BigModel)
prepare_fold(dataset, idx_train, idx_val, normalize="minmax", target_encoding="none",
             n_classes=None, out_range=None) -> (X_tr, y_tr, X_val, y_val, fitted: dict)
```
- `prepare_fold` es el **único** camino que usan runner y experimentos: ajusta el scaler **solo con `idx_train`** y devuelve `y` con forma `(n, k)` lista para `Network.fit`.

### `data/synthetic.py` (para configs de validación, F08)
`and_dataset()`, `xor_dataset()`, `line_samples(f, n, lo, hi, rng)` con las entradas del enunciado (±1).

## Tests y criterios de aceptación (`tests/test_data.py`)
> Los tests **nunca** dependen de `datasets/` (el CI no tiene los CSV): usan archivos sintéticos chicos creados en `tmp_path`. Si se agrega algún test con datos reales, se marca con `pytest.mark.skipif` cuando el archivo no existe.

- [x] `MinMaxScaler` lleva train exactamente a `[a, b]`; `inverse_transform(transform(X)) ≈ X`; columna constante → `a` con `warnings.warn`.
- [x] `StandardScaler`: media ≈ 0 y desvío ≈ 1 en train; desvío 0 → 0.
- [x] `UnitLengthScaler`: norma 2 de cada fila = 1; fila nula queda nula.
- [x] **Anti-leakage:** con valores extremos solo en validación, los estadísticos de `prepare_fold` no los ven.
- [x] `stratified_holdout` y `stratified_kfold`: proporción de clases en cada parte ±1 muestra de la global, incluido un caso 98/2.
- [x] `kfold`: folds de validación disjuntos, su unión es todo el dataset; reproducible con la semilla.
- [x] `one_hot` con `neg=-1`; forma `(n, k)`.
- [x] `load_csv` falla con mensaje claro ante columna inexistente o NaN (con `na_policy="error"`); `categorical` genera las columnas esperadas.
- [x] `synthetic`: AND y XOR iguales a los del enunciado (y a los de `tests/test_validation.py`).
- [x] `load_digits_csv` sobre un CSV sintético con el mismo formato (3 filas, `image` como texto): `X` de forma `(3, 784)` en [0, 1], `y` enteros, `image_shape == (28, 28)`; la segunda carga usa el caché y da lo mismo; si cambia el CSV, el caché se invalida.

## Entregables
Rama `f05-datos` lista para PR. El agente propone los mensajes de commit (`feat(F05): ...`) y no commitea sin permiso (`CLAUDE.md` §0).