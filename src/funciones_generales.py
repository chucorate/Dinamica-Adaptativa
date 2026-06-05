from typing import TYPE_CHECKING, cast
from functools import cached_property
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


@dataclass(frozen=True)
class TraitGrid:
    """
    Representa una malla cartesiana uniforme asociada a un espacio
    de rasgos potencialmente multidimensional.

    Esta estructura agrupa tanto la geometría de la malla como la
    información necesaria para reconstruir arreglos definidos sobre
    ella, evitando pasar múltiples objetos relacionados por separado.

    Attributes
    ----------
    points : np.ndarray
        Coordenadas de los puntos de la malla.

        Tiene forma

            (N_total, d),

        donde d es la dimensión del espacio de rasgos y N_total es el
        número total de puntos de la discretización.

        Cada fila contiene las coordenadas de un punto de la malla:

            points[k] = [x₁, x₂, ..., x_d].

    spacing : np.ndarray
        Vector de forma (d,) que contiene el paso espacial utilizado
        en cada dimensión de la malla.

    shape : tuple[int, ...]
        Forma multidimensional original de la malla antes de ser
        aplanada.

        Permite reconstruir arreglos definidos sobre la malla mediante
        operaciones como

            u.reshape(shape).

        Por ejemplo:

            d = 1  -> shape = (Nx,)
            d = 2  -> shape = (Nx, Ny)
            d = 3  -> shape = (Nx, Ny, Nz)
    """

    points: np.ndarray
    spacing: np.ndarray
    shape: tuple[int, ...]

    @property
    def size(self) -> int:
        return self.points.shape[0]

    @property
    def dimension(self) -> int:
        return self.points.shape[1]

    @cached_property
    def simpson_weights(self) -> np.ndarray:
        return get_simpson_weights(self.shape, self.spacing)


def _build_trait_grid(
    domain: np.ndarray,
    resolution: tuple[int, ...],
) -> TraitGrid:
    """
    Construye una malla cartesiana uniforme sobre un dominio
    multidimensional.
    """

    for N in resolution:
        assert N > 1 and isinstance(N, int)

    if len(domain) != len(resolution):
        raise ValueError("La dimensión del dominio y la resolución no coinciden.")

    axes = [np.linspace(a, b, N, dtype=dtype) for (a, b), N in zip(domain, resolution)]

    spacing = np.array([axis[1] - axis[0] for axis in axes], dtype=dtype)

    mesh = np.meshgrid(*axes, indexing="ij")

    shape = mesh[0].shape

    points = np.column_stack([m.ravel() for m in mesh])

    return TraitGrid(points=points, spacing=spacing, shape=shape)


def consumer_grid(model: "Model") -> TraitGrid:
    """
    Construye una malla cartesiana uniforme del espacio de rasgos
    de los consumidores.

    Si el espacio de rasgos tiene dimensión d_x y cada eje se
    discretiza con model.n_x puntos, la malla resultante contiene

        N_total = model.n_x ** d_x

    puntos.

    Returns
    -------
    grid : TraitGrid
        Grilla asociada al espacio multidimensional
        de los rasgos de consumidores.
    """

    return _build_trait_grid(model.consumer_domain, model.n_x)


def resource_grid(model: "Model") -> TraitGrid:
    """
    Construye una malla cartesiana uniforme del espacio de rasgos
    de los recursos.

    Si el espacio de rasgos tiene dimensión d_y y cada eje se
    discretiza con model.n_y puntos, la malla resultante contiene

        N_total = model.n_y ** d_y

    puntos.

    Returns
    -------
    grid : TraitGrid
        Grilla asociada al espacio multidimensional
        de los rasgos de recursos.
    """

    return _build_trait_grid(model.resource_domain, model.n_y)


def time_grid(model: "Model") -> tuple[np.ndarray, float]:
    """
    Construye la discretización temporal uniforme
    y devuelve también el tamaño del paso temporal.
    """
    t = np.linspace(0, model.T, model.n_t, dtype=dtype)
    delta_t = cast(float, t[1] - t[0])
    return t, delta_t


def _get_simpson_weights_1d(N: int, h: float) -> np.ndarray:
    """
    Genera un vector de pesos unidimensional de Simpson de tamaño N.

    Requiere N impar.
    """
    if N % 2 == 0:
        raise ValueError("La regla de Simpson requiere un número IMPAR de puntos (N).")

    weights = np.ones(N, dtype=dtype)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0

    return (h / 3.0) * weights


def get_simpson_weights(
    shape: tuple[int, ...],
    spacing: np.ndarray,
) -> np.ndarray:
    """
    Construye los pesos de Simpson asociados a una malla
    cartesiana multidimensional.

    Parameters
    ----------
    shape : tuple[int, ...]
        Forma de la malla.

    spacing : np.ndarray
        Paso espacial de cada dimensión.

    Returns
    -------
    np.ndarray
        Vector de forma

            (N_total,)

        compatible con TraitGrid.points.
    """

    weights = _get_simpson_weights_1d(shape[0], float(spacing[0]))

    for N, h in zip(shape[1:], spacing[1:]):
        weights = np.multiply.outer(
            weights,
            _get_simpson_weights_1d(N, float(h)),
        )

    return weights.ravel()


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
