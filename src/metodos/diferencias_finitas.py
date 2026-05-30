from typing import TYPE_CHECKING, cast, Literal

import numpy as np
from scipy.sparse import diags, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

if TYPE_CHECKING:
    from src.model import Model


def get_sparse_matrix(
    lamb: float,
    N_x: int,
    border_type: Literal["neumann", "periodic"],
) -> csc_matrix:
    diag_central = (1 + 2 * lamb) * np.ones(N_x)
    diag_lateral = -lamb * np.ones(N_x - 1)

    A = cast(
        lil_matrix,
        diags(
            [diag_lateral, diag_central, diag_lateral],  # type: ignore
            [-1, 0, 1],  # type: ignore
            format="lil",
            dtype=float,
        ).tolil(),
    )

    if border_type == "neumann":
        A[0, 1] = -2 * lamb  # u_{-1} = u_1
        A[-1, -2] = -2 * lamb  # u_{N} = u_{N-2}

    elif border_type == "periodic":
        A[0, -1] = -lamb  # vecino izquierdo del primer nodo = último nodo
        A[-1, 0] = -lamb  # vecino derecho del último nodo = primer nodo

    else:
        raise ValueError(f"Tipo de borde desconocido: {border_type}")

    return cast(csc_matrix, A.tocsc())


def compute_consumer_integral(
    kernel: np.ndarray, resource_distribution: np.ndarray, hy: float
) -> np.ndarray:
    """I_i = ∫ K(x_i, y) R(y) dy"""

    return hy * (kernel @ resource_distribution)


def compute_resource_integral(
    kernel: np.ndarray,
    growth_rate: np.ndarray,
    consumer_distribution: np.ndarray,
    hx: float,
) -> np.ndarray:
    """J_j = ∫ r(x) K(x, y_j) n(x) dx"""

    weighted_consumer = growth_rate * consumer_distribution

    return hx * (kernel.T @ weighted_consumer)


def solve_model_by_finite_differences(
    model: "Model",
    T: float,
    n_t: int,
    n_x: int,
    n_y: int,
    border_type: Literal["neumann", "periodic"],
) -> tuple[np.ndarray, np.ndarray]:
    # Discretización del espacio de rasgos, para consumer y resource.
    # De momento, se asume que solo hay un rasgo en consumer y resource domain
    x_min, x_max = cast(tuple[float, float], model.consumer_domain[0])
    x = np.linspace(x_min, x_max, n_x)
    hx = cast(float, x[1] - x[0])

    y_min, y_max = cast(tuple[float, float], model.resource_domain[0])
    y = np.linspace(y_min, y_max, n_y)
    hy = cast(float, y[1] - y[0])

    # Discretización del tiempo
    t = np.linspace(0, T, n_t)
    delta_t = cast(float, t[1] - t[0])

    # Inicializamos la matriz de consumer y resource
    consumer_distribution = np.zeros((n_t, n_x))
    consumer_distribution[0, :] = model.initial_consumer_distribution(x)

    resource_distribution = np.zeros((n_t, n_y))
    resource_distribution[0, :] = model.initial_resource_distribution(y)

    # Precalculamos parámetros del modelo que no dependen del tiempo
    lamb = model.mutation_rate * delta_t / (hx**2)
    A = get_sparse_matrix(lamb, n_x, border_type)

    kernel = model.resource_consumer_kernel(x[:, None], y[None, :])
    consumer_growth_rate = model.consumer_growth_rate(x)
    consumer_decay = model.consumer_decay(x)
    resouce_supply_rate = model.resource_supply_rate(y)
    resource_decay = model.resource_decay(y)

    for k in range(1, n_t):
        prev_consumer = consumer_distribution[k - 1, :]
        prev_resource = resource_distribution[k - 1, :]

        # Obtener consumer_distribution a tiempo k, resolvemos sistema de la forma Au = b
        consumer_integral = compute_consumer_integral(kernel, prev_resource, hy)

        b = prev_consumer * (
            1.0 + delta_t * (consumer_growth_rate * consumer_integral - consumer_decay)
        )
        consumer_distribution[k, :] = cast(np.ndarray, spsolve(A, b))

        # Obtener resource_distribution a tiempo k
        resource_integral = compute_resource_integral(
            kernel, consumer_growth_rate, prev_consumer, hx
        )

        resource_distribution[k, :] = prev_resource + delta_t * (
            resouce_supply_rate - prev_resource * (resource_decay + resource_integral)
        )

    return consumer_distribution, resource_distribution
