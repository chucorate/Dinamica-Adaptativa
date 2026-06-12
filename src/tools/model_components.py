from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np

from src.constants import dtype

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
        Matriz de interacción discreta de forma

            (N_consumer, N_resource),

        donde

            kernel[i,j] = K(x_i, y_j).

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
    """
    Evalúa y almacena todos los coeficientes funcionales del modelo.

    Se asumen vectores de la forma

        x.shape == (N_consumer, d_x)
        y.shape == (N_resource, d_y)

    En particular, el kernel debe retornar (N_consumer, N_resource).
    """

    return ModelCoefficients(
        kernel=model.resource_consumer_kernel(x, y).astype(dtype),
        consumer_growth_rate=model.consumer_growth_rate(x).astype(dtype),
        consumer_decay=model.consumer_decay(x).astype(dtype),
        resource_supply_rate=model.resource_supply_rate(y).astype(dtype),
        resource_decay=model.resource_decay(y).astype(dtype),
    )


def compute_consumer_integral(
    kernel: np.ndarray,
    resource_distribution: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Evalúa

        I(x_i) = ∫ K(x_i,y) R(y) dy

    sobre una malla multidimensional utilizando
    cuadratura de Simpson tensorial.

    Parameters
    ----------
    kernel : np.ndarray
        Matriz de interacción de forma

            (N_consumer, N_resource).

    resource_distribution : np.ndarray
        Distribución discreta del recurso de forma

            (N_resource,).

    weights : np.ndarray
        Pesos de Simpson asociados a la malla de recursos.

    Returns
    -------
    np.ndarray
        Aproximación de I evaluada en todos los
        puntos de la malla de consumidores.
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

        J(y_j) = ∫ r(x)K(x,y_j)n(x) dx

    sobre una malla multidimensional utilizando
    cuadratura de Simpson tensorial.

    Parameters
    ----------
    kernel : np.ndarray
        Matriz de interacción de forma

            (N_consumer, N_resource).

    growth_rate : np.ndarray
        Valores de r(x) evaluados en la malla
        de consumidores.

    consumer_distribution : np.ndarray
        Distribución discreta de consumidores.

    weights : np.ndarray
        Pesos de Simpson asociados a la malla
        de consumidores.

    Returns
    -------
    np.ndarray
        Aproximación de J evaluada en todos los
        puntos de la malla de recursos.
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
