# F06 · Métricas, curvas y selección de umbral

| | |
|---|---|
| **Objetivo** | Evaluación estándar de clasificadores binarios y multiclase, compatible con `Network.fit`, y herramientas para recomendar el umbral de fraude |
| **Requerimientos** | RF-10, RF-11 · `04-matematica.md` §6, §7 |
| **Depende de** | Nada del motor (se integra con `fit(metrics=...)` de F04, pero se puede testear sola) |
| **Estimación** | medio día con el agente |

## Especificación

### `core/metrics.py` — nivel etiquetas
```python
confusion_matrix(y_true, y_pred, n_classes=None) -> np.ndarray   # filas = real, columnas = predicho
binary_counts(y_true, y_pred, positive=1) -> dict[str, int]      # TP, FP, TN, FN
accuracy · precision · recall · f1 · fbeta(beta) · tpr · fpr     # binario, sobre etiquetas enteras
per_class_report(y_true, y_pred, n_classes) -> dict              # precision/recall/f1/support por clase
macro(report, metric) -> float
mse · rmse · mae                                                  # regresión / distillation
classification_summary(y_true, y_pred, n_classes) -> dict        # todo junto, serializable a JSON
```
- División por cero → 0.0 y la clave `"undefined": [...]` en el resumen (sin warnings ruidosos).

### `core/metrics.py` — nivel salidas de la red (para `fit` y el runner)
`Network.fit` recibe métricas `f(y_true, y_pred) -> float` con arrays `(n, n_salidas)` (targets codificados y salidas crudas). Adaptadores:
```python
to_labels(Y, *, task, threshold=0.5) -> np.ndarray
    # multiclass: argmax por fila (sirve para one-hot 0/1 y ±1)
    # binary: 1 si Y >= threshold (salida sigmoide) — para salida tanh usar threshold=0.0
get_metric(name, *, task: Literal["binary", "multiclass", "regression"], threshold=0.5) -> Metric
METRIC_NAMES = ("accuracy", "macro_f1", "precision", "recall", "f1", "f2", "mse", "mae")
```
Ejemplo: `net.fit(..., metrics={"accuracy": get_metric("accuracy", task="multiclass")})`.

### `core/thresholds.py`
```python
threshold_sweep(y_true, y_score, thresholds=np.linspace(0, 1, 101),
                cost_fn=1.0, cost_fp=1.0) -> pd.DataFrame
    # columnas: threshold, TP, FP, TN, FN, precision, recall, f1, f2, tpr, fpr, youden, cost
roc_curve(y_true, y_score) -> (fpr, tpr, thresholds)      # usa todos los scores únicos
pr_curve(y_true, y_score) -> (precision, recall, thresholds)
auc_trapezoid(x, y) -> float
average_precision(y_true, y_score) -> float
select_threshold(sweep_df, criterion: Literal["f1", "f2", "youden", "cost", "precision_at_recall"],
                 min_recall: float | None = None) -> dict   # {threshold, criterion, métricas en ese punto}
```

## Tests y criterios de aceptación
- [ ] **Ejemplo de la clase** (perros/gatos, 27 animales, matriz `[[11, 4], [2, 10]]`, perro = positivo): accuracy 21/27, precision 11/13, recall 11/15, FPR 2/12 y F1 exactos.
- [ ] `confusion_matrix` multiclase: suma = n, diagonal = aciertos, orientación filas = real verificada con un caso asimétrico.
- [ ] Macro-F1 en un caso de 3 clases armado a mano.
- [ ] Todo predicho negativo → precision `undefined`, sin excepción.
- [ ] `to_labels`: one-hot 0/1, one-hot ±1, salida sigmoide (umbral 0.5) y tanh (umbral 0).
- [ ] `get_metric("accuracy", task="multiclass")` funciona como `metric` de `Network.evaluate` sobre una red real chica.
- [ ] `auc_trapezoid`: scores perfectos → 1.0; invertidos → 0.0; constantes → 0.5; un caso chico a mano.
- [ ] `threshold_sweep`: en t = 0 recall = 1; en t > max(score) recall = 0.
- [ ] `select_threshold("cost")` con `cost_fn ≫ cost_fp` elige un umbral ≤ que con costos iguales.
- [ ] (Opcional) oráculo contra `sklearn.metrics` en datos aleatorios, solo en `tests/`.

## Entregables
Rama `f06-metricas` lista para PR. El agente propone los mensajes de commit (`feat(F06): ...`) y no commitea sin permiso (`CLAUDE.md` §0).