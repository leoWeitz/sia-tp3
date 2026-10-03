"""Barrido de umbral, curvas ROC y Precision-Recall, AUC y selección de umbral.

04-matematica §7. Problema binario: y_true con etiquetas 0/1 (1 = positivo,
fraude) de forma (n,) o (n, 1) e y_score (n,) o (n, 1), p. ej. la salida
sigmoide. Predicción positiva si score ≥ t. El umbral se elige en validación
y se reporta una sola vez en test (Constitución C5).
"""

from typing import Any, Literal

import numpy as np
import pandas as pd

Criterion = Literal["f1", "youden", "cost", "precision_at_recall"]
CRITERIA = ("f1", "youden", "cost", "precision_at_recall")
_MAXIMIZE = ("f1", "youden")

DEFAULT_THRESHOLDS = np.round(np.linspace(0.0, 1.0, 101), 2)


def _binary_inputs(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t, s = np.asarray(y_true), np.asarray(y_score, dtype=np.float64)
    t = t[:, 0] if t.ndim == 2 and t.shape[1] == 1 else t
    s = s[:, 0] if s.ndim == 2 and s.shape[1] == 1 else s
    if t.ndim != 1 or t.shape != s.shape:
        raise ValueError(f"y_true {t.shape} e y_score {s.shape} tienen que ser (n,) o (n, 1)")
    if not np.all(np.isin(t, (0, 1))):
        raise ValueError("y_true tiene que tener etiquetas 0/1 (1 = positivo)")
    return t.astype(bool), s


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den elemento a elemento, 0.0 donde den = 0 (sin warnings)."""
    num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    out = np.zeros(np.broadcast(num, den).shape)
    np.divide(num, den, out=out, where=den != 0)
    return out


def _counts_at(
    positive: np.ndarray, y_score: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """TP(t) y FP(t) para cada umbral (score ≥ t) con búsqueda binaria: O((n + T) log n)."""
    pos = np.sort(y_score[positive])
    neg = np.sort(y_score[~positive])
    tp = len(pos) - np.searchsorted(pos, thresholds, side="left")
    fp = len(neg) - np.searchsorted(neg, thresholds, side="left")
    return tp, fp, len(pos), len(neg)


def threshold_sweep(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray = DEFAULT_THRESHOLDS,
    cost_fn: float = 1.0,
    cost_fp: float = 1.0,
) -> pd.DataFrame:
    """Una fila por umbral t (en orden creciente) con las métricas de predecir score ≥ t.

    Columnas: threshold, TP, FP, TN, FN, precision, recall, f1, tpr, fpr,
    youden (= tpr − fpr) y cost (= cost_fn·FN + cost_fp·FP). División por cero
    → 0.0.
    """
    positive, s = _binary_inputs(y_true, y_score)
    th = np.sort(np.asarray(thresholds, dtype=np.float64))
    tp, fp, n_pos, n_neg = _counts_at(positive, s, th)
    fn, tn = n_pos - tp, n_neg - fp
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, n_pos + np.zeros_like(tp))
    fpr_ = _safe_div(fp, n_neg + np.zeros_like(fp))
    return pd.DataFrame(
        {
            "threshold": th,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "precision": prec,
            "recall": rec,
            "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
            "tpr": rec,
            "fpr": fpr_,
            "youden": rec - fpr_,
            "cost": cost_fn * fn + cost_fp * fp,
        }
    )


def _curve_points(y_true: np.ndarray, y_score: np.ndarray):
    """Conteos en cada score único (decreciente), con un primer punto en t = +inf."""
    positive, s = _binary_inputs(y_true, y_score)
    thresholds = np.concatenate([[np.inf], np.unique(s)[::-1]])
    tp, fp, n_pos, n_neg = _counts_at(positive, s, thresholds)
    return tp, fp, n_pos, n_neg, thresholds


def roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fpr, tpr, thresholds) usando todos los scores únicos como umbral.

    thresholds es decreciente y empieza en +inf (punto (0, 0)); el último
    umbral es el score mínimo (punto (1, 1)). Hacen falta positivos y negativos.
    """
    tp, fp, n_pos, n_neg, thresholds = _curve_points(y_true, y_score)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("La curva ROC necesita al menos un positivo y un negativo")
    return fp / n_neg, tp / n_pos, thresholds


def pr_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(precision, recall, thresholds) usando todos los scores únicos como umbral.

    thresholds es decreciente (recall creciente) y empieza en +inf con el punto
    convencional precision = 1, recall = 0. Hace falta al menos un positivo.
    """
    tp, fp, n_pos, _, thresholds = _curve_points(y_true, y_score)
    if n_pos == 0:
        raise ValueError("La curva PR necesita al menos un positivo")
    prec = _safe_div(tp, tp + fp)
    prec[0] = 1.0
    return prec, tp / n_pos, thresholds


def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    """Área bajo la curva y(x) por trapecios, ordenando los puntos por x (§7)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"x {x.shape} e y {y.shape} tienen que ser vectores del mismo largo")
    order = np.lexsort((y, x))
    x, y = x[order], y[order]
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2.0))


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AP = Σ_k (R_k − R_{k−1}) P_k sobre los puntos de pr_curve (§7), sin interpolar."""
    prec, rec, _ = pr_curve(y_true, y_score)
    return float(np.sum(np.diff(rec) * prec[1:]))


def _to_python(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def select_threshold(
    sweep_df: pd.DataFrame,
    criterion: Criterion,
    min_recall: float | None = None,
) -> dict[str, Any]:
    """Elige el umbral de un threshold_sweep según un criterio (§7).

    "f1", "youden": máximo · "cost": mínimo costo (los costos se fijan en
    threshold_sweep) · "precision_at_recall": máxima precision con recall ≥
    min_recall. Ante empates se queda con el menor umbral. Devuelve la fila
    elegida como dict JSON ({threshold, criterion, TP, ..., cost}, más
    min_recall si corresponde).
    """
    if criterion not in CRITERIA:
        raise ValueError(f"Criterio desconocido: {criterion!r}. Opciones: {list(CRITERIA)}")
    df = sweep_df.sort_values("threshold", kind="stable").reset_index(drop=True)
    if criterion in _MAXIMIZE:
        idx = int(df[criterion].to_numpy().argmax())
    elif criterion == "cost":
        idx = int(df["cost"].to_numpy().argmin())
    else:
        if min_recall is None:
            raise ValueError("criterion='precision_at_recall' necesita min_recall")
        if not 0.0 <= min_recall <= 1.0:
            raise ValueError(f"min_recall tiene que estar en [0, 1], llegó {min_recall}")
        ok = df["recall"].to_numpy() >= min_recall
        if not ok.any():
            raise ValueError(f"Ningún umbral del barrido alcanza recall ≥ {min_recall}")
        precisions = np.where(ok, df["precision"].to_numpy(), -np.inf)
        idx = int(precisions.argmax())
    row = {k: _to_python(v) for k, v in df.iloc[idx].to_dict().items()}
    for k in ("TP", "FP", "TN", "FN"):
        row[k] = int(row[k])
    result: dict[str, Any] = {"threshold": row.pop("threshold"), "criterion": criterion, **row}
    if criterion == "precision_at_recall":
        result["min_recall"] = min_recall
    return result
