"""Métricas de clasificación y regresión.

04-matematica §6. Dos niveles:
- Etiquetas: y_true, y_pred enteros (n,) o (n, 1). Matriz de confusión con
  filas = clase real y columnas = clase predicha (convención de la cátedra).
  Las métricas binarias toman `positive` como clase positiva (default 1).
- Salidas de la red: get_metric arma una Metric f(y_true, y_pred) -> float
  sobre arrays (n, n_salidas) (targets codificados y salidas crudas), que es lo
  que reciben Network.fit y Network.evaluate. to_labels hace la conversión.

División por cero → 0.0, sin warnings; los resúmenes listan en "undefined"
qué métricas cayeron en ese caso.
"""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

Metric = Callable[[np.ndarray, np.ndarray], float]
Task = Literal["binary", "multiclass", "regression"]

TASKS = ("binary", "multiclass", "regression")
METRIC_NAMES = ("accuracy", "macro_f1", "precision", "recall", "f1", "f2", "mse", "mae")
PER_CLASS_METRICS = ("precision", "recall", "f1")


def _labels(y: np.ndarray, name: str) -> np.ndarray:
    """Etiquetas enteras (n,) desde (n,) o (n, 1)."""
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1:
        raise ValueError(f"{name} tiene que ser (n,) o (n, 1) con etiquetas, llegó {y.shape}")
    if y.size and not np.all(np.mod(y, 1) == 0):
        raise ValueError(f"{name} tiene que tener etiquetas enteras")
    return y.astype(np.int64)


def _pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t, p = _labels(y_true, "y_true"), _labels(y_pred, "y_pred")
    if t.shape != p.shape:
        raise ValueError(f"y_true {t.shape} y y_pred {p.shape} tienen que tener la misma forma")
    return t, p


def _div(num: float, den: float) -> tuple[float, bool]:
    """(num / den, definido). Con den = 0 devuelve (0.0, False)."""
    if den == 0:
        return 0.0, False
    return float(num) / float(den), True


# --- Nivel etiquetas ---


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None
) -> np.ndarray:
    """Matriz (n_classes, n_classes) de conteos: filas = real, columnas = predicho.

    Etiquetas enteras en [0, n_classes); None = max etiqueta + 1.
    """
    t, p = _pair(y_true, y_pred)
    if t.size and min(t.min(), p.min()) < 0:
        raise ValueError("Las etiquetas tienen que ser enteros ≥ 0")
    top = int(max(t.max(), p.max())) + 1 if t.size else 0
    if n_classes is None:
        n_classes = top
    elif top > n_classes:
        raise ValueError(f"Hay etiquetas ≥ n_classes = {n_classes}")
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (t, p), 1)
    return cm


def binary_counts(y_true: np.ndarray, y_pred: np.ndarray, positive: int = 1) -> dict[str, int]:
    """{"TP", "FP", "TN", "FN"} tomando `positive` como clase positiva."""
    t, p = _pair(y_true, y_pred)
    tp, pp = t == positive, p == positive
    return {
        "TP": int(np.sum(tp & pp)),
        "FP": int(np.sum(~tp & pp)),
        "TN": int(np.sum(~tp & ~pp)),
        "FN": int(np.sum(tp & ~pp)),
    }


def _fbeta_counts(tp: int, fp: int, fn: int, beta: float) -> tuple[float, bool]:
    """F_β = (1+β²)PR / (β²P + R) = (1+β²)TP / ((1+β²)TP + β²FN + FP) (§6).

    La forma con conteos evita dividir por cero cuando P o R no están
    definidas; solo queda indefinida con TP = FP = FN = 0.
    """
    b2 = beta**2
    return _div((1 + b2) * tp, (1 + b2) * tp + b2 * fn + fp)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Aciertos / total (binario y multiclase: traza / total)."""
    t, p = _pair(y_true, y_pred)
    return _div(np.sum(t == p), t.size)[0]


def precision(y_true: np.ndarray, y_pred: np.ndarray, positive: int = 1) -> float:
    """TP / (TP + FP); 0.0 si no hay positivos predichos."""
    c = binary_counts(y_true, y_pred, positive)
    return _div(c["TP"], c["TP"] + c["FP"])[0]


def recall(y_true: np.ndarray, y_pred: np.ndarray, positive: int = 1) -> float:
    """TP / (TP + FN) (= TPR); 0.0 si no hay positivos reales."""
    c = binary_counts(y_true, y_pred, positive)
    return _div(c["TP"], c["TP"] + c["FN"])[0]


def tpr(y_true: np.ndarray, y_pred: np.ndarray, positive: int = 1) -> float:
    """Tasa de verdaderos positivos, igual a recall."""
    return recall(y_true, y_pred, positive)


def fpr(y_true: np.ndarray, y_pred: np.ndarray, positive: int = 1) -> float:
    """FP / (FP + TN); 0.0 si no hay negativos reales."""
    c = binary_counts(y_true, y_pred, positive)
    return _div(c["FP"], c["FP"] + c["TN"])[0]


def fbeta(y_true: np.ndarray, y_pred: np.ndarray, beta: float, positive: int = 1) -> float:
    """F_β (β > 1 prioriza recall); 0.0 si TP = FP = FN = 0."""
    c = binary_counts(y_true, y_pred, positive)
    return _fbeta_counts(c["TP"], c["FP"], c["FN"], beta)[0]


def f1(y_true: np.ndarray, y_pred: np.ndarray, positive: int = 1) -> float:
    """F1 = 2PR / (P + R)."""
    return fbeta(y_true, y_pred, 1.0, positive)


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, Any]:
    """Métricas uno contra todos por clase.

    Devuelve {"precision", "recall", "f1": listas de n_classes floats,
    "support": lista de ints (muestras reales por clase), "undefined": nombres
    "metrica[clase]" que dividieron por cero}. Serializable a JSON.
    """
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    report: dict[str, Any] = {m: [] for m in PER_CLASS_METRICS}
    undefined: list[str] = []
    for c in range(n_classes):
        values = {
            "precision": _div(tp[c], tp[c] + fp[c]),
            "recall": _div(tp[c], tp[c] + fn[c]),
            "f1": _fbeta_counts(tp[c], fp[c], fn[c], 1.0),
        }
        for m, (value, ok) in values.items():
            report[m].append(value)
            if not ok:
                undefined.append(f"{m}[{c}]")
    report["support"] = cm.sum(axis=1).astype(int).tolist()
    report["undefined"] = undefined
    return report


def macro(report: dict[str, Any], metric: str) -> float:
    """Promedio simple entre clases de una métrica de per_class_report (§6)."""
    if metric not in PER_CLASS_METRICS:
        raise ValueError(
            f"Métrica por clase desconocida: {metric!r}. Opciones: {PER_CLASS_METRICS}"
        )
    values = report[metric]
    return float(np.mean(values)) if values else 0.0


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Error cuadrático medio sobre todas las muestras y salidas (igual que core.losses.MSE)."""
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Raíz de mse."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Error absoluto medio sobre todas las muestras y salidas."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def classification_summary(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> dict[str, Any]:
    """Todo junto, serializable a JSON (para metrics.json y final_eval.json).

    Siempre: n, accuracy, confusion_matrix, per_class, macro_precision,
    macro_recall, macro_f1, undefined. Con n_classes = 2 agrega las binarias con
    positivo = 1: TP, FP, TN, FN, precision, recall, f1, f2, tpr, fpr.
    """
    cm = confusion_matrix(y_true, y_pred, n_classes)
    report = per_class_report(y_true, y_pred, n_classes)
    summary: dict[str, Any] = {
        "n": int(cm.sum()),
        "accuracy": _div(np.trace(cm), cm.sum())[0],
        "confusion_matrix": cm.tolist(),
        "per_class": {k: v for k, v in report.items() if k != "undefined"},
        **{f"macro_{m}": macro(report, m) for m in PER_CLASS_METRICS},
    }
    undefined: list[str] = []
    if n_classes == 2:
        c = binary_counts(y_true, y_pred)
        tp, fp, tn, fn = c["TP"], c["FP"], c["TN"], c["FN"]
        binary = {
            "precision": _div(tp, tp + fp),
            "recall": _div(tp, tp + fn),
            "f1": _fbeta_counts(tp, fp, fn, 1.0),
            "f2": _fbeta_counts(tp, fp, fn, 2.0),
            "tpr": _div(tp, tp + fn),
            "fpr": _div(fp, fp + tn),
        }
        summary.update(c)
        for m, (value, ok) in binary.items():
            summary[m] = value
            if not ok:
                undefined.append(m)
    summary["undefined"] = undefined + report["undefined"]
    return summary


# --- Nivel salidas de la red ---

_LABEL_METRICS: dict[str, Metric] = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "f2": lambda t, p: fbeta(t, p, 2.0),
}
_BINARY_ONLY = ("precision", "recall", "f1", "f2")


def _check_task(task: str) -> None:
    if task not in TASKS:
        raise ValueError(f"task desconocida: {task!r}. Opciones: {list(TASKS)}")


def to_labels(Y: np.ndarray, *, task: Task, threshold: float = 0.5) -> np.ndarray:
    """Salidas o targets de la red (n, k) → etiquetas enteras (n,).

    multiclass: argmax por fila (sirve para one-hot 0/1 y ±1), k ≥ 2.
    binary: 1 si Y ≥ threshold, con Y (n,) o (n, 1); threshold = 0.5 para
    salida sigmoide y 0.0 para tanh.
    """
    _check_task(task)
    Y = np.asarray(Y)
    if task == "multiclass":
        if Y.ndim != 2 or Y.shape[1] < 2:
            raise ValueError(f"multiclass necesita Y (n, k) con k ≥ 2, llegó {Y.shape}")
        return Y.argmax(axis=1)
    if task == "binary":
        if Y.ndim == 2 and Y.shape[1] == 1:
            Y = Y[:, 0]
        if Y.ndim != 1:
            raise ValueError(f"binary necesita Y (n,) o (n, 1), llegó {Y.shape}")
        return (Y >= threshold).astype(np.int64)
    raise ValueError("task='regression' no tiene etiquetas")


def get_metric(name: str, *, task: Task, threshold: float = 0.5) -> Metric:
    """Metric f(y_true, y_pred) -> float para Network.fit / evaluate.

    y_true son los targets codificados y y_pred las salidas crudas, ambos
    (n, n_salidas); se pasan a etiquetas con to_labels (el mismo threshold para
    los dos en binario). mse y mae se calculan sobre los valores crudos y
    valen para cualquier task. precision, recall, f1 y f2 son binarias
    (positivo = 1); en multiclase usar macro_f1.
    """
    _check_task(task)
    if name not in METRIC_NAMES:
        raise ValueError(f"Métrica desconocida: {name!r}. Opciones: {list(METRIC_NAMES)}")
    if name in ("mse", "mae"):
        fn: Metric = mse if name == "mse" else mae
    elif task == "regression":
        raise ValueError(f"La métrica {name!r} es de clasificación y task='regression'")
    elif task == "multiclass" and name in _BINARY_ONLY:
        raise ValueError(f"{name!r} es binaria; en multiclase usar 'macro_f1' o 'accuracy'")
    else:

        def fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            t = to_labels(y_true, task=task, threshold=threshold)
            p = to_labels(y_pred, task=task, threshold=threshold)
            if name == "macro_f1":
                n_classes = np.asarray(y_true).shape[1] if task == "multiclass" else 2
                return macro(per_class_report(t, p, n_classes), "f1")
            return _LABEL_METRICS[name](t, p)

    def metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(fn(y_true, y_pred))

    metric.__name__ = name
    return metric
