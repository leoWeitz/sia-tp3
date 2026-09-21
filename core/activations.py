from typing import Protocol
import numpy as np


class Activation(Protocol):

    def forward(self, z: np.ndarray) -> np.ndarray:
        """Aplica la activación a la pre-activación z."""
        ...

    def backward(self, z: np.ndarray) -> np.ndarray:
        """Derivada de la activación evaluada en z, misma forma que z."""
        ...


class Identity:
    """f(z) = z. Salida de un perceptrón simple lineal."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        return z

    def backward(self, z: np.ndarray) -> np.ndarray:
        return np.ones_like(z)


class Tanh:
    """f(z) = tanh(beta * z), con imagen en (-1, 1).

    beta controla la pendiente: valores altos hacen la transición más abrupta.
    Con beta = 1 es la tangente hiperbólica clásica.
    """

    def __init__(self, beta: float = 1.0) -> None:
        self.beta = beta

    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(self.beta * z)

    def backward(self, z: np.ndarray) -> np.ndarray:
        # d/dz tanh(beta*z) = beta * (1 - tanh(beta*z)^2)
        return self.beta * (1.0 - np.tanh(self.beta * z) ** 2)