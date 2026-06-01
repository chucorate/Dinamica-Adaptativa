from typing import TYPE_CHECKING, cast

import numpy as np

dtype = np.float64

if TYPE_CHECKING:
    from src.model import Model


def consumer_grid(model: "Model") -> tuple[np.ndarray, float]:
    """
    Construye la malla uniforme del espacio de rasgos
    de los consumidores y devuelve también su paso espacial.
    """
    xmin, xmax = model.consumer_domain[0]
    x = np.linspace(xmin, xmax, model.n_x, dtype=dtype)
    hx = cast(float, x[1] - x[0])
    return x, hx


def resource_grid(model: "Model") -> tuple[np.ndarray, float]:
    """
    Construye la malla uniforme del espacio de rasgos
    de los recursos y devuelve también su paso espacial.
    """
    ymin, ymax = model.resource_domain[0]
    y = np.linspace(ymin, ymax, model.n_y, dtype=dtype)
    hy = cast(float, y[1] - y[0])
    return y, hy


def time_grid(model: "Model") -> tuple[np.ndarray, float]:
    """
    Construye la discretización temporal uniforme
    y devuelve también el tamaño del paso temporal.
    """
    t = np.linspace(0, model.T, model.n_t, dtype=dtype)
    delta_t = cast(float, t[1] - t[0])
    return t, delta_t


def get_simpson_weights(N: int, h: float) -> np.ndarray:
    """
    Genera un vector de pesos de Simpson de tamaño N.

    Requiere N impar.
    """
    if N % 2 == 0:
        raise ValueError("La regla de Simpson requiere un número IMPAR de puntos (N).")

    weights = np.ones(N, dtype=dtype)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0

    return (h / 3.0) * weights


def compute_consumer_integral(
    kernel: np.ndarray,
    resource_distribution: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Evalúa

        I(x_i) = ∫ K(x_i,y) R(y) dy

    mediante cuadratura numérica.
    """
    weighted_resource = weights * resource_distribution
    return kernel @ weighted_resource


def compute_resource_integral(
    kernel: np.ndarray,
    growth_rate: np.ndarray,
    consumer_distribution: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Evalúa

        J(y_j) = ∫ r(x) K(x,y_j) n(x) dx

    mediante cuadratura numérica.
    """
    weighted_consumer = weights * growth_rate * consumer_distribution
    return kernel.T @ weighted_consumer


def compute_stationary_resource(
    kernel: np.ndarray,
    consumer_growth_rate: np.ndarray,
    consumer_distribution: np.ndarray,
    weights_x: np.ndarray,
    resource_supply_rate: np.ndarray,
    resource_decay: np.ndarray,
) -> np.ndarray:
    """
    Calcula el recurso estacionario

        R(y) = Rin(y) / ( m2(y) + ∫ r(x)K(x,y)n(x)dx )

    """

    resource_integral = compute_resource_integral(
        kernel, consumer_growth_rate, consumer_distribution, weights_x
    )

    return resource_supply_rate / (resource_decay + resource_integral)
