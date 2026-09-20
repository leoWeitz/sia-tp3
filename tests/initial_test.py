# tests/test_setup.py
import numpy as np


def test_numpy_disponible():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert (x @ x.T).shape == (2, 2)


def test_paquetes_importables():
    import analysis  # noqa: F401
    import core  # noqa: F401
    import data  # noqa: F401
    import experiments  # noqa: F401
