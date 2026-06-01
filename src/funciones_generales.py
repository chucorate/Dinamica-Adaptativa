from typing import TYPE_CHECKING, cast
from dataclasses import dataclass

import numpy as np

dtype = np.float64

if TYPE_CHECKING:
    from src.model import Model


@dataclass(frozen=True)
class ModelCoefficients:
    """
    Agrupa los coeficientes espaciales del modelo evaluados sobre
    las mallas de consumidores y recursos.

    Su propósito es evitar reevaluar funciones durante la simulación y
    reducir el número de argumentos que deben recibir los algoritmos
    numéricos.

    Attributes
    ----------
    kernel : np.ndarray
        Matriz de interacción K(x_i, y_j) entre consumidores y recursos.

    consumer_growth_rate : np.ndarray
        Tasa de crecimiento r(x_i) de cada rasgo de consumidor.

    consumer_decay : np.ndarray
        Tasa de mortalidad o decaimiento m₁(x_i) de cada rasgo de consumidor.

    resource_supply_rate : np.ndarray
        Tasa de aporte externo R_in(y_j) de cada rasgo de recurso.

    resource_decay : np.ndarray
        Tasa de decaimiento natural m₂(y_j) de cada rasgo de recurso.
    """

    kernel: np.ndarray
    consumer_growth_rate: np.ndarray
    consumer_decay: np.ndarray
    resource_supply_rate: np.ndarray
    resource_decay: np.ndarray


def build_model_coefficients(
    model: "Model",
    x: np.ndarray,
    y: np.ndarray,
) -> ModelCoefficients:
    """Evalúa y almacena todos los coeficientes funcionales del modelo."""

    return ModelCoefficients(
        kernel=model.resource_consumer_kernel(x[:, None], y[None, :]).astype(dtype),
        consumer_growth_rate=model.consumer_growth_rate(x).astype(dtype),
        consumer_decay=model.consumer_decay(x).astype(dtype),
        resource_supply_rate=model.resource_supply_rate(y).astype(dtype),
        resource_decay=model.resource_decay(y).astype(dtype),
    )


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
    consumer_distribution: np.ndarray,
    weights_x: np.ndarray,
    coeffs: ModelCoefficients,
) -> np.ndarray:
    """
    Calcula el recurso estacionario

        R(y) = Rin(y) / ( m2(y) + ∫ r(x)K(x,y)n(x)dx )

    """

    resource_integral = compute_resource_integral(
        coeffs.kernel, coeffs.consumer_growth_rate, consumer_distribution, weights_x
    )

    return coeffs.resource_supply_rate / (coeffs.resource_decay + resource_integral)
