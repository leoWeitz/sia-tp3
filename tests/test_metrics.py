"""Tests de F06: métricas (core/metrics.py) y umbral (core/thresholds.py)."""

import json

import numpy as np
import pytest

from core.losses import MSE
from core.metrics import (
    METRIC_NAMES,
    accuracy,
    binary_counts,
    classification_summary,
    confusion_matrix,
    f1,
    fbeta,
    fpr,
    get_metric,
    macro,
    mae,
    mse,
    per_class_report,
    precision,
    recall,
    rmse,
    to_labels,
    tpr,
)
from core.network import Network
from core.optimizers import GD
from core.thresholds import (
    auc_trapezoid,
    average_precision,
    pr_curve,
    roc_curve,
    select_threshold,
    threshold_sweep,
)


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def labels_from_matrix(cm) -> tuple[np.ndarray, np.ndarray]:
    """Etiquetas (y_true, y_pred) que reproducen la matriz cm (filas = real)."""
    y_true, y_pred = [], []
    for real, row in enumerate(cm):
        for pred, count in enumerate(row):
            y_true += [real] * count
            y_pred += [pred] * count
    return np.array(y_true), np.array(y_pred)


# --- Ejemplo de la clase: perros (positivo) y gatos ---
# Matriz de la clase con el positivo primero: [[TP, FN], [FP, TN]] = [[11, 4], [2, 10]].
# Con perro = 1 y gato = 0, en nuestra convención (índice = clase) es [[10, 2], [4, 11]].
DOG_TRUE, DOG_PRED = labels_from_matrix([[10, 2], [4, 11]])


def test_ejemplo_de_la_clase_perros_y_gatos():
    assert len(DOG_TRUE) == 27
    assert binary_counts(DOG_TRUE, DOG_PRED) == {"TP": 11, "FP": 2, "TN": 10, "FN": 4}
    assert accuracy(DOG_TRUE, DOG_PRED) == pytest.approx(21 / 27)
    assert precision(DOG_TRUE, DOG_PRED) == pytest.approx(11 / 13)
    assert recall(DOG_TRUE, DOG_PRED) == pytest.approx(11 / 15)
    assert tpr(DOG_TRUE, DOG_PRED) == pytest.approx(11 / 15)
    assert fpr(DOG_TRUE, DOG_PRED) == pytest.approx(2 / 12)
    p, r = 11 / 13, 11 / 15
    assert f1(DOG_TRUE, DOG_PRED) == pytest.approx(2 * p * r / (p + r))
    assert fbeta(DOG_TRUE, DOG_PRED, beta=2.0) == pytest.approx(5 * p * r / (4 * p + r))
    assert fbeta(DOG_TRUE, DOG_PRED, beta=1.0) == pytest.approx(f1(DOG_TRUE, DOG_PRED))


def test_positive_configurable():
    # Tomando gato (0) como positivo se intercambian los roles.
    assert binary_counts(DOG_TRUE, DOG_PRED, positive=0) == {"TP": 10, "FP": 4, "TN": 11, "FN": 2}


def test_acepta_columnas_n_1():
    assert accuracy(DOG_TRUE[:, None], DOG_PRED[:, None]) == pytest.approx(21 / 27)


def test_formas_distintas_es_error():
    with pytest.raises(ValueError):
        accuracy(np.array([0, 1]), np.array([0, 1, 1]))


# --- Matriz de confusión ---


def test_confusion_matrix_multiclase_orientacion():
    y_true = np.array([0, 0, 0, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 0, 2])
    cm = confusion_matrix(y_true, y_pred)
    # Asimétrico: dos 0 predichos como 1 (fila 0, col 1) y un 2 predicho como 0 (fila 2, col 0).
    np.testing.assert_array_equal(cm, [[1, 2, 0], [0, 1, 0], [1, 0, 1]])
    assert cm.sum() == len(y_true)
    assert np.trace(cm) == np.sum(y_true == y_pred)
    assert accuracy(y_true, y_pred) == pytest.approx(np.trace(cm) / cm.sum())


def test_confusion_matrix_n_classes_explicito_y_errores():
    cm = confusion_matrix(np.array([0, 1]), np.array([1, 1]), n_classes=4)
    assert cm.shape == (4, 4)
    with pytest.raises(ValueError):
        confusion_matrix(np.array([0, 4]), np.array([0, 1]), n_classes=4)
    with pytest.raises(ValueError):
        confusion_matrix(np.array([0, -1]), np.array([0, 1]))
    with pytest.raises(ValueError):
        confusion_matrix(np.array([0.5, 1.0]), np.array([0, 1]))


# --- Multiclase: reporte por clase y macro ---


def test_macro_f1_tres_clases_a_mano():
    # Filas = real. Clase 0: TP 3, FN 1, FP 1 · Clase 1: TP 2, FN 2, FP 1
    # · Clase 2: TP 1, FN 0, FP 1
    cm = [[3, 1, 0], [1, 2, 1], [0, 0, 1]]
    y_true, y_pred = labels_from_matrix(cm)
    report = per_class_report(y_true, y_pred, n_classes=3)
    np.testing.assert_allclose(report["precision"], [3 / 4, 2 / 3, 1 / 2])
    np.testing.assert_allclose(report["recall"], [3 / 4, 2 / 4, 1 / 1])
    f1s = [3 / 4, 2 * (2 / 3) * (1 / 2) / (2 / 3 + 1 / 2), 2 * 0.5 * 1 / 1.5]
    np.testing.assert_allclose(report["f1"], f1s)
    assert report["support"] == [4, 4, 1]
    assert macro(report, "f1") == pytest.approx(np.mean(f1s))
    assert macro(report, "recall") == pytest.approx((0.75 + 0.5 + 1) / 3)


def test_macro_metrica_desconocida():
    report = per_class_report(np.array([0, 1]), np.array([0, 1]), n_classes=2)
    with pytest.raises(ValueError):
        macro(report, "support")


# --- División por cero ---


def test_todo_predicho_negativo_precision_undefined_sin_excepcion():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.zeros(4, dtype=int)
    with np.errstate(all="raise"):
        assert precision(y_true, y_pred) == 0.0
        assert f1(y_true, y_pred) == 0.0
        summary = classification_summary(y_true, y_pred, n_classes=2)
    assert summary["precision"] == 0.0
    assert "precision" in summary["undefined"]
    assert "precision[1]" in summary["undefined"]
    assert "recall" not in summary["undefined"]


def test_sin_positivos_reales_recall_undefined():
    y_true = np.zeros(3, dtype=int)
    summary = classification_summary(y_true, np.array([0, 1, 0]), n_classes=2)
    assert summary["recall"] == 0.0
    assert "recall" in summary["undefined"]


def test_classification_summary_serializable_y_completo():
    summary = classification_summary(DOG_TRUE, DOG_PRED, n_classes=2)
    json.dumps(summary)
    assert summary["n"] == 27
    assert summary["accuracy"] == pytest.approx(21 / 27)
    assert summary["confusion_matrix"] == [[10, 2], [4, 11]]
    assert summary["precision"] == pytest.approx(11 / 13)
    assert summary["fpr"] == pytest.approx(2 / 12)
    assert summary["undefined"] == []
    assert set(summary) >= {"macro_precision", "macro_recall", "macro_f1", "per_class", "f2"}


def test_classification_summary_multiclase_sin_claves_binarias():
    y = np.array([0, 1, 2, 2])
    summary = classification_summary(y, y, n_classes=3)
    json.dumps(summary)
    assert summary["macro_f1"] == 1.0
    assert "precision" not in summary


# --- Regresión ---


def test_mse_rmse_mae():
    y_true = np.array([[0.0], [1.0], [2.0]])
    y_pred = np.array([[1.0], [1.0], [0.0]])
    assert mse(y_true, y_pred) == pytest.approx(5 / 3)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(5 / 3))
    assert mae(y_true, y_pred) == pytest.approx(1.0)
    assert mse(y_true, y_pred) == pytest.approx(MSE().value(y_true, y_pred))


# --- Nivel salidas de la red ---


def test_to_labels_one_hot_01_y_pm1():
    Y01 = np.array([[0.1, 0.8, 0.1], [0.9, 0.0, 0.2]])
    np.testing.assert_array_equal(to_labels(Y01, task="multiclass"), [1, 0])
    Ypm = np.array([[-1.0, -1.0, 1.0], [1.0, -1.0, -1.0]])
    np.testing.assert_array_equal(to_labels(Ypm, task="multiclass"), [2, 0])


def test_to_labels_binario_sigmoide_y_tanh():
    s = np.array([[0.2], [0.5], [0.7]])
    np.testing.assert_array_equal(to_labels(s, task="binary"), [0, 1, 1])
    t = np.array([[-0.3], [0.0], [0.4]])
    np.testing.assert_array_equal(to_labels(t, task="binary", threshold=0.0), [0, 1, 1])


def test_to_labels_errores():
    with pytest.raises(ValueError):
        to_labels(np.zeros((3, 1)), task="multiclass")
    with pytest.raises(ValueError):
        to_labels(np.zeros((3, 2)), task="binary")
    with pytest.raises(ValueError):
        to_labels(np.zeros((3, 1)), task="regression")


def test_get_metric_con_salidas_crudas():
    y_true = np.eye(3)[[0, 1, 2, 2]]  # one-hot de [0, 1, 2, 2]
    y_pred = np.array([[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.1, 0.8, 0.1], [0.0, 0.3, 0.6]])
    assert get_metric("accuracy", task="multiclass")(y_true, y_pred) == pytest.approx(0.75)
    macro_f1 = get_metric("macro_f1", task="multiclass")(y_true, y_pred)
    assert macro_f1 == pytest.approx(np.mean([1.0, 2 / 3, 2 / 3]))

    t = np.array([[1.0], [-1.0], [1.0], [-1.0]])
    o = np.array([[0.4], [-0.2], [-0.1], [0.3]])
    assert get_metric("recall", task="binary", threshold=0.0)(t, o) == pytest.approx(0.5)
    assert get_metric("precision", task="binary", threshold=0.0)(t, o) == pytest.approx(0.5)
    assert get_metric("f2", task="binary", threshold=0.0)(t, o) == pytest.approx(0.5)
    assert get_metric("mse", task="regression")(t, o) == pytest.approx(mse(t, o))


def test_get_metric_nombres_y_errores():
    assert set(METRIC_NAMES) == {
        "accuracy",
        "macro_f1",
        "precision",
        "recall",
        "f1",
        "f2",
        "mse",
        "mae",
    }
    with pytest.raises(ValueError, match="desconocida"):
        get_metric("auc", task="binary")
    with pytest.raises(ValueError, match="task"):
        get_metric("accuracy", task="clustering")
    with pytest.raises(ValueError):
        get_metric("accuracy", task="regression")
    with pytest.raises(ValueError, match="macro_f1"):
        get_metric("f1", task="multiclass")


def test_get_metric_con_network_evaluate_y_fit():
    X = rng(1).normal(size=(30, 4))
    labels = np.arange(30) % 3
    Y = np.eye(3)[labels]
    net = Network([4, 5, 3], output_activation="sigmoid", rng=rng(2))
    acc = get_metric("accuracy", task="multiclass")
    result = net.evaluate(X, Y, MSE(), metric=acc)
    expected = np.mean(net.predict(X).argmax(axis=1) == labels)
    assert result["metric"] == pytest.approx(expected)

    history = net.fit(
        X,
        Y,
        GD(lr=0.5),
        MSE(),
        epochs=3,
        metrics={"accuracy": acc, "macro_f1": get_metric("macro_f1", task="multiclass")},
    )
    assert len(history.column("train_accuracy")) == 3
    assert all(0.0 <= v <= 1.0 for v in history.column("train_macro_f1"))


# --- Umbral: barrido, ROC, PR, AUC ---


def test_auc_perfecto_invertido_constante():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    assert auc_trapezoid(*roc_curve(y, s)[:2]) == pytest.approx(1.0)
    assert auc_trapezoid(*roc_curve(y, 1 - s)[:2]) == pytest.approx(0.0)
    assert auc_trapezoid(*roc_curve(y, np.full(6, 0.4))[:2]) == pytest.approx(0.5)


def test_auc_trapezoid_a_mano():
    assert auc_trapezoid(np.array([0.0, 0.5, 1.0]), np.array([0.0, 1.0, 1.0])) == pytest.approx(
        0.75
    )
    # Desordenado: se ordena por x.
    assert auc_trapezoid(np.array([1.0, 0.0, 0.5]), np.array([1.0, 0.0, 1.0])) == pytest.approx(
        0.75
    )


def test_roc_caso_chico_a_mano():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    fpr_, tpr_, th = roc_curve(y, s)
    # Umbrales: +inf, 0.8, 0.4, 0.35, 0.1
    np.testing.assert_allclose(th[1:], [0.8, 0.4, 0.35, 0.1])
    np.testing.assert_allclose(fpr_, [0.0, 0.0, 0.0, 0.5, 1.0])
    np.testing.assert_allclose(tpr_, [0.0, 0.5, 1.0, 1.0, 1.0])
    assert auc_trapezoid(fpr_, tpr_) == pytest.approx(1.0)
    s2 = np.array([0.1, 0.35, 0.4, 0.8])  # un negativo por encima de un positivo
    assert auc_trapezoid(*roc_curve(y, s2)[:2]) == pytest.approx(0.75)


def test_roc_sin_positivos_es_error():
    with pytest.raises(ValueError):
        roc_curve(np.zeros(3), np.array([0.1, 0.2, 0.3]))


def test_pr_curve_y_average_precision_a_mano():
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1])
    p, r, th = pr_curve(y, s)
    np.testing.assert_allclose(th[1:], [0.9, 0.8, 0.7, 0.1])
    np.testing.assert_allclose(r, [0.0, 0.5, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(p, [1.0, 1.0, 0.5, 2 / 3, 0.5])
    assert average_precision(y, s) == pytest.approx(0.5 * 1.0 + 0.5 * 2 / 3)


def test_threshold_sweep_extremos():
    y = rng(3).integers(0, 2, size=50)
    s = rng(4).uniform(0.05, 0.9, size=50)
    df = threshold_sweep(y, s)
    expected_cols = ["threshold", "TP", "FP", "TN", "FN", "precision", "recall", "f1", "f2",
                     "tpr", "fpr", "youden", "cost"]  # fmt: skip
    assert list(df.columns) == expected_cols
    assert len(df) == 101
    assert df.iloc[0]["threshold"] == 0.0
    assert df.iloc[0]["recall"] == 1.0
    last = df[df["threshold"] > s.max()]
    assert len(last) > 0 and (last["recall"] == 0.0).all()
    assert ((df["TP"] + df["FP"] + df["TN"] + df["FN"]) == 50).all()


def test_threshold_sweep_coincide_con_metricas_de_etiquetas():
    y = rng(5).integers(0, 2, size=40)
    s = rng(6).uniform(size=40)
    df = threshold_sweep(y, s, thresholds=np.array([0.3, 0.5]), cost_fn=5.0, cost_fp=1.0)
    for _, row in df.iterrows():
        pred = (s >= row["threshold"]).astype(int)
        assert row["precision"] == pytest.approx(precision(y, pred))
        assert row["f2"] == pytest.approx(fbeta(y, pred, beta=2.0))
        assert row["youden"] == pytest.approx(tpr(y, pred) - fpr(y, pred))
        counts = binary_counts(y, pred)
        assert row["cost"] == pytest.approx(5 * counts["FN"] + counts["FP"])


def cost_case():
    # Positivos con scores medios: bajar el umbral los atrapa a costa de falsos positivos.
    y = np.array([0] * 20 + [1] * 5)
    s = np.concatenate([np.linspace(0.0, 0.6, 20), [0.3, 0.45, 0.55, 0.7, 0.9]])
    return y, s


def test_select_threshold_costo_fn_alto_elige_umbral_menor_o_igual():
    y, s = cost_case()
    t_equal = select_threshold(threshold_sweep(y, s), "cost")["threshold"]
    t_fn = select_threshold(threshold_sweep(y, s, cost_fn=50.0, cost_fp=1.0), "cost")["threshold"]
    assert t_fn <= t_equal
    assert t_fn < t_equal  # en este caso es estrictamente menor


@pytest.mark.parametrize("criterion", ["f1", "f2", "youden"])
def test_select_threshold_maximiza(criterion):
    y, s = cost_case()
    df = threshold_sweep(y, s)
    chosen = select_threshold(df, criterion)
    json.dumps(chosen)
    assert chosen["criterion"] == criterion
    assert chosen[criterion] == pytest.approx(df[criterion].max())


def test_select_threshold_precision_at_recall():
    y, s = cost_case()
    df = threshold_sweep(y, s)
    chosen = select_threshold(df, "precision_at_recall", min_recall=0.8)
    assert chosen["recall"] >= 0.8
    assert chosen["precision"] == pytest.approx(df[df["recall"] >= 0.8]["precision"].max())
    assert chosen["min_recall"] == 0.8
    with pytest.raises(ValueError, match="min_recall"):
        select_threshold(df, "precision_at_recall")
    with pytest.raises(ValueError):
        select_threshold(df, "precision_at_recall", min_recall=1.5)
    with pytest.raises(ValueError, match="desconocido"):
        select_threshold(df, "accuracy")


# --- Oráculo: sklearn (opcional, solo en tests) ---


def test_oraculo_sklearn():
    skm = pytest.importorskip("sklearn.metrics")
    r = rng(7)
    for _ in range(5):
        y_true = r.integers(0, 4, size=80)
        y_pred = np.where(r.uniform(size=80) < 0.6, y_true, r.integers(0, 4, size=80))
        np.testing.assert_array_equal(
            confusion_matrix(y_true, y_pred, 4),
            skm.confusion_matrix(y_true, y_pred, labels=range(4)),
        )
        report = per_class_report(y_true, y_pred, 4)
        assert macro(report, "f1") == pytest.approx(
            skm.f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
        yb, pb = (y_true == 1).astype(int), (y_pred == 1).astype(int)
        assert precision(yb, pb) == pytest.approx(skm.precision_score(yb, pb, zero_division=0))
        assert recall(yb, pb) == pytest.approx(skm.recall_score(yb, pb, zero_division=0))
        assert fbeta(yb, pb, 2.0) == pytest.approx(
            skm.fbeta_score(yb, pb, beta=2.0, zero_division=0)
        )

        scores = np.round(r.uniform(size=80), 2)  # con empates
        assert auc_trapezoid(*roc_curve(yb, scores)[:2]) == pytest.approx(
            skm.roc_auc_score(yb, scores)
        )
        assert average_precision(yb, scores) == pytest.approx(
            skm.average_precision_score(yb, scores)
        )
