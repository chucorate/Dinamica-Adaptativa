from typing import TYPE_CHECKING, cast
from functools import cached_property
from dataclasses import dataclass

import numpy as np

from src.constants import dtype

if TYPE_CHECKING:
    from src.model import Model


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
    delta_t = float(t[1] - t[0])
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
