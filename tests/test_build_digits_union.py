"""Tests de data/build_digits_union.py con CSV sintéticos chicos."""

import hashlib

import pytest

from data.build_digits_union import HEADER, build_union, main, pixel_key, read_rows, to_csv_bytes


def _row(label: int, value: float, fmt: str = "{:.1f}") -> bytes:
    pixels = ", ".join(fmt.format(value) for _ in range(784))
    return f'{label},"[{pixels}]"'.encode()


def _write(path, rows, newline=b"\r\n"):
    path.write_bytes(HEADER + newline + newline.join(rows) + newline)


def test_read_rows_acepta_crlf_y_lf(tmp_path):
    rows = [_row(1, 0.0), _row(2, 0.5)]
    _write(tmp_path / "crlf.csv", rows, b"\r\n")
    _write(tmp_path / "lf.csv", rows, b"\n")
    assert read_rows(tmp_path / "crlf.csv") == rows
    assert read_rows(tmp_path / "lf.csv") == rows


def test_read_rows_rechaza_otro_encabezado(tmp_path):
    (tmp_path / "x.csv").write_bytes(b"a,b\n1,2\n")
    with pytest.raises(ValueError):
        read_rows(tmp_path / "x.csv")


def test_pixel_key_ignora_el_formato_de_los_numeros():
    assert pixel_key(_row(3, 0.5)) == pixel_key(_row(3, 0.5, "{:.3f}"))
    assert pixel_key(_row(3, 0.5)) != pixel_key(_row(3, 0.25))


def test_build_union_saca_repetidas_y_conserva_el_orden():
    a, b, c = _row(0, 0.0), _row(1, 0.5), _row(2, 1.0)
    rows, info = build_union([a, b], [b, c])
    assert rows == [a, b, c]
    assert info["duplicates"] == 1
    assert info["label_conflicts"] == 0


def test_build_union_cuenta_conflictos_de_etiqueta():
    rows, info = build_union([_row(0, 0.5)], [_row(7, 0.5)])
    assert rows == [_row(0, 0.5)]
    assert info["label_conflicts"] == 1


def test_to_csv_bytes_usa_lf():
    data = to_csv_bytes([_row(0, 0.0)])
    assert b"\r\n" not in data
    assert data.startswith(HEADER + b"\n") and data.endswith(b"\n")


def test_main_escribe_la_union_y_detecta_fuga(tmp_path, capsys):
    a, b, c, t = _row(0, 0.0), _row(1, 0.5), _row(2, 1.0), _row(3, 0.25)
    _write(tmp_path / "digits.csv", [a, b])
    _write(tmp_path / "more_digits.csv", [b, c])
    _write(tmp_path / "digits_test.csv", [t])
    # Con datos sintéticos el sha256 no es el de results/ej3_*, así que devuelve 1,
    # pero el archivo se escribe igual.
    assert main(["--datasets-dir", str(tmp_path)]) == 1
    out = (tmp_path / "digits_union.csv").read_bytes()
    assert out == to_csv_bytes([a, b, c])
    assert hashlib.sha256(out).hexdigest() in capsys.readouterr().out

    _write(tmp_path / "digits_test.csv", [c])
    assert main(["--datasets-dir", str(tmp_path)]) == 1
    assert "ERROR" in capsys.readouterr().out
