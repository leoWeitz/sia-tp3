"""Tests de F05: carga, preprocesamiento y particiones (data/).

Nunca dependen de datasets/ (el CI no tiene los CSV): usan archivos
sintéticos chicos creados en tmp_path.
"""

import json
import warnings

import numpy as np
import pytest

from data.loaders import Dataset, load_csv, load_digits_csv
from data.preprocess import (
    SCALERS,
    IdentityScaler,
    MinMaxScaler,
    StandardScaler,
    TargetScaler,
    UnitLengthScaler,
    get_scaler,
    one_hot,
    scaler_from_state_dict,
)
from data.splits import (
    holdout,
    kfold,
    prepare_fold,
    stratified_holdout,
    stratified_kfold,
    stratify_labels,
)
from data.synthetic import and_dataset, line_samples, xor_dataset


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- Scalers ---


@pytest.mark.parametrize("feature_range", [(0.0, 1.0), (-1.0, 1.0), (0.1, 0.9)])
def test_minmax_lleva_train_exactamente_al_rango(feature_range):
    a, b = feature_range
    X = rng().normal(3.0, 5.0, size=(40, 4))
    scaler = MinMaxScaler(feature_range).fit(X)
    Xs = scaler.transform(X)
    np.testing.assert_allclose(Xs.min(axis=0), a)
    np.testing.assert_allclose(Xs.max(axis=0), b)
    np.testing.assert_allclose(scaler.inverse_transform(Xs), X)


def test_minmax_columna_constante_va_a_a_y_avisa():
    X = np.column_stack([np.linspace(0, 1, 5), np.full(5, 7.0)])
    with pytest.warns(UserWarning, match="constante"):
        scaler = MinMaxScaler((-1.0, 1.0)).fit(X)
    Xs = scaler.transform(X)
    np.testing.assert_array_equal(Xs[:, 1], -1.0)
    assert np.all(np.isfinite(Xs))
    np.testing.assert_allclose(scaler.inverse_transform(Xs), X)


def test_minmax_rango_invalido_es_error():
    with pytest.raises(ValueError):
        MinMaxScaler((1.0, 1.0))


def test_standard_media_cero_desvio_uno():
    X = rng(1).uniform(-10, 50, size=(60, 3))
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    np.testing.assert_allclose(Xs.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Xs.std(axis=0), 1.0)
    np.testing.assert_allclose(scaler.inverse_transform(Xs), X)


def test_standard_desvio_cero_da_cero():
    X = np.column_stack([np.arange(5.0), np.full(5, 3.0)])
    Xs = StandardScaler().fit(X).transform(X)
    np.testing.assert_array_equal(Xs[:, 1], 0.0)
    assert np.all(np.isfinite(Xs))


def test_unit_length_norma_uno_y_fila_nula_queda_nula():
    X = rng(2).normal(size=(6, 5))
    X[3] = 0.0
    Xs = UnitLengthScaler().fit(X).transform(X)
    norms = np.linalg.norm(Xs, axis=1)
    np.testing.assert_allclose(np.delete(norms, 3), 1.0)
    np.testing.assert_array_equal(Xs[3], 0.0)


def test_identity_no_cambia_y_devuelve_copia():
    X = rng().normal(size=(4, 2))
    scaler = IdentityScaler().fit(X)
    Xs = scaler.transform(X)
    np.testing.assert_array_equal(Xs, X)
    assert Xs is not X


@pytest.mark.parametrize("cls", [MinMaxScaler, StandardScaler])
def test_transform_sin_fit_es_error(cls):
    with pytest.raises(RuntimeError, match="fit"):
        cls().transform(np.ones((2, 2)))


@pytest.mark.parametrize("name", ["none", "minmax", "zscore", "unit_length"])
def test_state_dict_ida_y_vuelta(name):
    X = rng(3).normal(size=(10, 3))
    scaler = get_scaler(name).fit(X)
    state = scaler.state_dict()
    restored = scaler_from_state_dict(state)
    assert type(restored) is type(scaler)
    np.testing.assert_array_equal(restored.transform(X), scaler.transform(X))


def test_get_scaler_nombres_y_errores():
    assert set(SCALERS) == {"none", "minmax", "zscore", "unit_length"}
    assert get_scaler("minmax", feature_range=(-1, 1)).feature_range == (-1.0, 1.0)
    with pytest.raises(ValueError, match="desconocido"):
        get_scaler("robust")
    with pytest.raises(TypeError):
        get_scaler("zscore", feature_range=(0, 1))


def test_target_scaler_ajusta_a_imagen_de_salida_e_invierte():
    y = rng(4).uniform(-3, 8, size=(30, 1))
    ts = TargetScaler((-1.0, 1.0)).fit(y)
    ys = ts.transform(y)
    assert ys.min() == pytest.approx(-1.0)
    assert ys.max() == pytest.approx(1.0)
    np.testing.assert_allclose(ts.inverse_transform(ys), y)


def test_target_scaler_con_rango_de_entrada_fijo_es_identidad_para_probabilidades():
    y = rng(5).uniform(0.1, 0.4, size=(20, 1))  # probabilidades que no llegan a 0 ni a 1
    ts = TargetScaler((0.0, 1.0), in_range=(0.0, 1.0)).fit(y)
    np.testing.assert_allclose(ts.transform(y), y)


# --- one_hot ---


def test_one_hot_forma_y_valores():
    y = np.array([2, 0, 1, 2])
    Y = one_hot(y, 3)
    assert Y.shape == (4, 3)
    np.testing.assert_array_equal(Y.argmax(axis=1), y)
    np.testing.assert_array_equal(Y.sum(axis=1), 1.0)


def test_one_hot_con_neg_menos_uno():
    Y = one_hot(np.array([[1], [0]]), 3, neg=-1.0)
    np.testing.assert_array_equal(Y, [[-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]])


def test_one_hot_etiqueta_fuera_de_rango_es_error():
    with pytest.raises(ValueError):
        one_hot(np.array([0, 3]), 3)


# --- Particiones ---


def assert_particion(idx_train, idx_val, n):
    assert len(np.intersect1d(idx_train, idx_val)) == 0
    np.testing.assert_array_equal(np.union1d(idx_train, idx_val), np.arange(n))


def test_holdout_tamanos_disjuntos_y_reproducible():
    tr, va = holdout(50, 0.8, rng(7))
    assert len(tr) == 40 and len(va) == 10
    assert_particion(tr, va, 50)
    tr2, va2 = holdout(50, 0.8, rng(7))
    np.testing.assert_array_equal(tr, tr2)
    np.testing.assert_array_equal(va, va2)


@pytest.mark.parametrize("ratio", [0.0, 1.0, 1.5])
def test_holdout_ratio_invalido(ratio):
    with pytest.raises(ValueError):
        holdout(10, ratio, rng())


def assert_proporciones(y, parts):
    """Cada clase en cada parte está a ±1 muestra de la proporción global."""
    classes, counts = np.unique(y, return_counts=True)
    for idx in parts:
        for c, count in zip(classes, counts, strict=True):
            expected = count / len(y) * len(idx)
            assert abs(np.sum(y[idx] == c) - expected) <= 1, (c, len(idx))


@pytest.mark.parametrize("counts", [(50, 50), (98, 2), (980, 20), (30, 25, 17, 9, 1)], ids=str)
@pytest.mark.parametrize("ratio", [0.8, 0.7])
def test_stratified_holdout_mantiene_proporciones(counts, ratio):
    y = rng(1).permutation(np.repeat(np.arange(len(counts)), counts))
    tr, va = stratified_holdout(y, ratio, rng(2))
    assert_particion(tr, va, len(y))
    assert len(tr) == round(ratio * len(y))
    assert_proporciones(y, [tr, va])


def test_stratified_holdout_acepta_y_columna():
    y = np.repeat([0, 1], [8, 2]).reshape(-1, 1)
    tr, va = stratified_holdout(y, 0.5, rng())
    assert_particion(tr, va, 10)
    assert np.sum(y[va] == 1) == 1


@pytest.mark.parametrize("n, k", [(20, 5), (23, 4), (7, 7)])
def test_kfold_disjuntos_union_total_y_reproducible(n, k):
    folds = kfold(n, k, rng(3))
    assert len(folds) == k
    vals = [va for _, va in folds]
    np.testing.assert_array_equal(np.sort(np.concatenate(vals)), np.arange(n))
    for tr, va in folds:
        assert_particion(tr, va, n)
    sizes = [len(va) for va in vals]
    assert max(sizes) - min(sizes) <= 1
    for (tr1, va1), (tr2, va2) in zip(folds, kfold(n, k, rng(3)), strict=True):
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(va1, va2)
    assert any(
        not np.array_equal(va1, va2)
        for (_, va1), (_, va2) in zip(folds, kfold(n, k, rng(4)), strict=True)
    )


def test_kfold_k_invalido():
    with pytest.raises(ValueError):
        kfold(5, 1, rng())
    with pytest.raises(ValueError):
        kfold(5, 6, rng())


@pytest.mark.parametrize("counts", [(50, 50), (98, 2), (490, 10), (40, 33, 12, 5)], ids=str)
@pytest.mark.parametrize("k", [2, 5])
def test_stratified_kfold_mantiene_proporciones(counts, k):
    y = rng(5).permutation(np.repeat(np.arange(len(counts)), counts))
    folds = stratified_kfold(y, k, rng(6))
    vals = [va for _, va in folds]
    np.testing.assert_array_equal(np.sort(np.concatenate(vals)), np.arange(len(y)))
    sizes = [len(va) for va in vals]
    assert max(sizes) - min(sizes) <= 1
    for tr, va in folds:
        assert_particion(tr, va, len(y))
        assert_proporciones(y, [tr, va])


def test_stratify_labels_bins_por_cuantiles():
    y = np.linspace(0, 1, 100) ** 4  # sesgado hacia 0, como las probabilidades de fraude
    labels = stratify_labels(y, n_bins=10)
    assert labels.shape == (100,)
    _, counts = np.unique(labels, return_counts=True)
    assert len(counts) == 10
    assert np.all(counts == 10)
    assert np.all(np.diff(labels[np.argsort(y)]) >= 0)  # monótono en y


def test_stratify_labels_con_empates_no_rompe():
    y = np.concatenate([np.zeros(90), np.linspace(0.5, 1, 10)])
    labels = stratify_labels(y, n_bins=10)
    assert len(np.unique(labels)) >= 2
    tr, va = stratified_holdout(labels, 0.8, rng())
    assert_particion(tr, va, 100)


# --- prepare_fold ---


def make_dataset(X, y) -> Dataset:
    return Dataset(
        X=np.asarray(X, dtype=float),
        y=np.asarray(y),
        feature_names=[f"f{i}" for i in range(np.shape(X)[1])],
        target_name="t",
        meta={},
    )


def test_prepare_fold_anti_leakage():
    X = rng(8).uniform(0, 1, size=(20, 3))
    idx_train, idx_val = np.arange(15), np.arange(15, 20)
    X[idx_val] = [1e6, -1e6, 500.0]  # extremos solo en validación
    ds = make_dataset(X, np.zeros(20))
    X_tr, _, X_val, _, fitted = prepare_fold(ds, idx_train, idx_val, normalize="minmax")
    scaler = fitted["scaler"]
    np.testing.assert_array_equal(scaler.data_min, X[idx_train].min(axis=0))
    np.testing.assert_array_equal(scaler.data_max, X[idx_train].max(axis=0))
    np.testing.assert_allclose(X_tr.min(axis=0), 0.0)
    np.testing.assert_allclose(X_tr.max(axis=0), 1.0)
    assert X_val.max() > 1.0  # validación queda fuera del rango: no se ajustó con ella

    _, _, _, _, fitted_z = prepare_fold(ds, idx_train, idx_val, normalize="zscore")
    np.testing.assert_allclose(fitted_z["scaler"].mean, X[idx_train].mean(axis=0))


def test_prepare_fold_no_modifica_el_dataset():
    X = rng(9).normal(size=(10, 2))
    ds = make_dataset(X.copy(), np.arange(10) % 2)
    prepare_fold(ds, np.arange(8), np.arange(8, 10), normalize="zscore", target_encoding="onehot")
    np.testing.assert_array_equal(ds.X, X)


def test_prepare_fold_formas_y_encodings():
    y = np.array([0, 1, 2, 1, 0, 2, 1, 0])
    ds = make_dataset(rng().normal(size=(8, 2)), y)
    tr, va = np.arange(6), np.arange(6, 8)

    _, y_tr, _, y_val, fitted = prepare_fold(ds, tr, va, target_encoding="none")
    assert y_tr.shape == (6, 1) and y_val.shape == (2, 1)
    assert fitted["target_scaler"] is None

    _, y_tr, _, y_val, fitted = prepare_fold(ds, tr, va, target_encoding="onehot")
    assert y_tr.shape == (6, 3) and y_val.shape == (2, 3)
    assert fitted["n_classes"] == 3
    np.testing.assert_array_equal(y_tr.argmax(axis=1), y[tr])

    _, y_tr, _, _, _ = prepare_fold(ds, tr, va, target_encoding="pm1_onehot", n_classes=4)
    assert y_tr.shape == (6, 4)
    assert set(np.unique(y_tr)) == {-1.0, 1.0}


def test_prepare_fold_scale_to_output_ajusta_solo_con_train():
    y = np.concatenate([np.linspace(0, 10, 8), [100.0, -100.0]])
    ds = make_dataset(rng().normal(size=(10, 2)), y)
    tr, va = np.arange(8), np.arange(8, 10)
    _, y_tr, _, y_val, fitted = prepare_fold(
        ds, tr, va, target_encoding="scale_to_output", out_range=(-1.0, 1.0)
    )
    np.testing.assert_allclose([y_tr.min(), y_tr.max()], [-1.0, 1.0])
    assert abs(y_val).max() > 1.0
    np.testing.assert_allclose(fitted["target_scaler"].inverse_transform(y_tr), y[tr, None])
    with pytest.raises(ValueError, match="out_range"):
        prepare_fold(ds, tr, va, target_encoding="scale_to_output")


@pytest.mark.parametrize("encoding", ["none", "onehot", "pm1_onehot", "scale_to_output"])
def test_prepare_fold_sin_validacion(encoding):
    # split "none" (Ej1 con todas las muestras): idx_val vacío.
    ds = make_dataset(rng().normal(size=(6, 2)), np.array([0, 1, 2, 1, 0, 2]))
    X_tr, y_tr, X_val, y_val, _ = prepare_fold(
        ds, np.arange(6), np.array([], dtype=np.int64), normalize="zscore",
        target_encoding=encoding, out_range=(0.0, 1.0),
    )  # fmt: skip
    assert X_tr.shape == (6, 2) and X_val.shape == (0, 2)
    assert y_val.shape == (0, y_tr.shape[1])


def test_prepare_fold_opciones_invalidas():
    ds = make_dataset(np.ones((4, 1)), np.array([0, 1, 0, 1]))
    with pytest.raises(ValueError):
        prepare_fold(ds, np.arange(2), np.arange(2, 4), target_encoding="ordinal")
    with pytest.raises(ValueError):
        prepare_fold(ds, np.arange(3), np.arange(2, 4))  # se solapan


# --- load_csv ---


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_load_csv_basico(tmp_path):
    path = write(tmp_path, "d.csv", "id,a,b,t\n1,0.5,2,0.1\n2,1.5,3,0.9\n")
    ds = load_csv(path, target="t", drop=["id"])
    assert ds.feature_names == ["a", "b"]
    assert ds.X.shape == (2, 2) and ds.X.dtype == np.float64
    np.testing.assert_array_equal(ds.y, [0.1, 0.9])
    assert ds.target_name == "t"
    assert len(ds.meta["sha256"]) == 64
    assert ds.meta["n_rows"] == 2


def test_load_csv_features_explicitas(tmp_path):
    path = write(tmp_path, "d.csv", "a,b,c,t\n1,2,3,0\n4,5,6,1\n")
    ds = load_csv(path, target="t", features=["c", "a"])
    assert ds.feature_names == ["c", "a"]
    np.testing.assert_array_equal(ds.X, [[3, 1], [6, 4]])


def test_load_csv_columna_inexistente(tmp_path):
    path = write(tmp_path, "d.csv", "a,t\n1,0\n")
    with pytest.raises(ValueError, match="'zz'"):
        load_csv(path, target="zz")
    with pytest.raises(ValueError, match="'b'"):
        load_csv(path, target="t", features=["b"])
    with pytest.raises(ValueError, match="'b'"):
        load_csv(path, target="t", drop=["b"])


def test_load_csv_nan_es_error_por_columna(tmp_path):
    path = write(tmp_path, "d.csv", "a,b,t\n1,,0\n2,3,1\n,4,0\n,5,1\n")
    with pytest.raises(ValueError, match=r"a.*2") as exc:
        load_csv(path, target="t")
    assert "b" in str(exc.value)


def test_load_csv_drop_rows_registra_en_meta(tmp_path):
    path = write(tmp_path, "d.csv", "a,b,t\n1,,0\n2,3,1\n4,4,0\n")
    ds = load_csv(path, target="t", na_policy="drop_rows")
    assert ds.X.shape == (2, 2)
    assert ds.meta["dropped_rows"] == 1
    assert ds.meta["dropped_row_indices"] == [0]


def test_load_csv_nan_en_columna_descartada_no_importa(tmp_path):
    path = write(tmp_path, "d.csv", "a,notes,t\n1,,0\n2,x,1\n")
    ds = load_csv(path, target="t", drop=["notes"])
    assert ds.X.shape == (2, 1)


def test_load_csv_categorical_one_hot(tmp_path):
    path = write(tmp_path, "d.csv", "a,color,t\n1,red,0\n2,blue,1\n3,red,0\n")
    ds = load_csv(path, target="t", categorical=["color"])
    assert ds.feature_names == ["a", "color=blue", "color=red"]
    np.testing.assert_array_equal(ds.X, [[1, 0, 1], [2, 1, 0], [3, 0, 1]])
    assert ds.meta["encoded_columns"] == {"color": ["blue", "red"]}


def test_load_csv_no_numerica_sin_categorical_es_error(tmp_path):
    path = write(tmp_path, "d.csv", "a,color,t\n1,red,0\n")
    with pytest.raises(ValueError, match="color"):
        load_csv(path, target="t")


def test_load_csv_na_policy_desconocida(tmp_path):
    path = write(tmp_path, "d.csv", "a,t\n1,0\n")
    with pytest.raises(ValueError):
        load_csv(path, target="t", na_policy="impute")


# --- load_digits_csv ---


def write_digits(path, labels, seed=0):
    images = rng(seed).uniform(0, 1, size=(len(labels), 784)).round(4)
    lines = ["label,image"]
    for label, img in zip(labels, images, strict=True):
        lines.append(f'{label},"{json.dumps(img.tolist())}"')
    path.write_text("\n".join(lines) + "\n")
    return images


def test_load_digits_csv_formato_y_cache(tmp_path):
    path = tmp_path / "digits.csv"
    images = write_digits(path, [3, 0, 9])
    ds = load_digits_csv(path)
    assert ds.X.shape == (3, 784) and ds.X.dtype == np.float64
    assert ds.X.min() >= 0.0 and ds.X.max() <= 1.0
    np.testing.assert_array_equal(ds.X, images)
    np.testing.assert_array_equal(ds.y, [3, 0, 9])
    assert np.issubdtype(ds.y.dtype, np.integer)
    assert ds.meta["image_shape"] == (28, 28)
    assert ds.meta["n_rows"] == 3
    assert ds.feature_names[0] == "px_0" and ds.feature_names[-1] == "px_783"
    assert ds.meta["from_cache"] is False
    cache = tmp_path / "digits.npz"
    assert cache.exists()

    ds2 = load_digits_csv(path)
    assert ds2.meta["from_cache"] is True
    np.testing.assert_array_equal(ds2.X, ds.X)
    np.testing.assert_array_equal(ds2.y, ds.y)
    assert ds2.meta["sha256"] == ds.meta["sha256"]

    images_new = write_digits(path, [1, 2, 3], seed=1)  # cambia el CSV
    ds3 = load_digits_csv(path)
    assert ds3.meta["from_cache"] is False
    np.testing.assert_array_equal(ds3.X, images_new)
    np.testing.assert_array_equal(ds3.y, [1, 2, 3])


def test_load_digits_csv_sin_cache_no_escribe(tmp_path):
    path = tmp_path / "d.csv"
    write_digits(path, [5])
    load_digits_csv(path, cache=False)
    assert not (tmp_path / "d.npz").exists()


def test_load_digits_csv_imagen_de_largo_incorrecto(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text('label,image\n1,"[0.1, 0.2]"\n')
    with pytest.raises(ValueError, match="784"):
        load_digits_csv(path, cache=False)


# --- synthetic ---

# Mismas entradas que tests/test_validation.py (esquinas de [−1, 1]²).
X_LOGIC = np.array([[-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [1.0, 1.0]])


def test_and_y_xor_del_enunciado():
    ds_and, ds_xor = and_dataset(), xor_dataset()
    np.testing.assert_array_equal(ds_and.X, X_LOGIC)
    np.testing.assert_array_equal(ds_xor.X, X_LOGIC)
    np.testing.assert_array_equal(ds_and.y, [[-1.0], [-1.0], [-1.0], [1.0]])
    np.testing.assert_array_equal(ds_xor.y, [[1.0], [1.0], [-1.0], [-1.0]])


def test_and_y_xor_coinciden_con_la_logica():
    a, b = X_LOGIC[:, 0] > 0, X_LOGIC[:, 1] > 0
    np.testing.assert_array_equal(and_dataset().y[:, 0] > 0, a & b)
    np.testing.assert_array_equal(xor_dataset().y[:, 0] > 0, a ^ b)


def test_line_samples():
    ds = line_samples(np.tanh, 50, -2.0, 2.0, rng(0))
    assert ds.X.shape == (50, 1) and ds.y.shape == (50, 1)
    assert ds.X.min() >= -2.0 and ds.X.max() <= 2.0
    np.testing.assert_allclose(ds.y, np.tanh(ds.X))
    np.testing.assert_array_equal(line_samples(np.tanh, 50, -2.0, 2.0, rng(0)).X, ds.X)


def test_no_hay_warnings_en_caso_normal():
    X = rng().normal(size=(10, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MinMaxScaler().fit(X).transform(X)
