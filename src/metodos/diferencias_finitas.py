from typing import TYPE_CHECKING, cast, Literal

import numpy as np
from scipy.sparse import diags, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

if TYPE_CHECKING:
    from src.model import Model


dtype = np.float64


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
            dtype=dtype,
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


def _get_simpson_weights(N: int, h: float) -> np.ndarray:
    """Genera un vector de pesos de Simpson de tamaño N.

    Requiere que N sea impar."""
    if N % 2 == 0:
        raise ValueError("La regla de Simpson requiere un número IMPAR de puntos (N).")

    weights = np.ones(N, dtype=dtype)
    weights[1:-1:2] = 4.0  # Índices impares reciben peso 4
    weights[2:-1:2] = 2.0  # Índices pares (excepto extremos) reciben peso 2

    return (h / 3.0) * weights


def compute_consumer_integral(
    kernel: np.ndarray,
    resource_distribution: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """I_i = ∫ K(x_i, y) R(y) dy"""

    weighted_resource = weights * resource_distribution

    return kernel @ weighted_resource


def compute_resource_integral(
    kernel: np.ndarray,
    growth_rate: np.ndarray,
    consumer_distribution: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """J_j = ∫ r(x) K(x, y_j) n(x) dx"""

    weighted_consumer = weights * growth_rate * consumer_distribution

    return kernel.T @ weighted_consumer


def solve_model_by_finite_differences(
    model: "Model",
    T: float,
    n_t: int,
    n_x: int,
    n_y: int,
    border_type: Literal["neumann", "periodic"],
    use_stationary_resource: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # 1. Discretización del dominio
    x_min, x_max = cast(tuple[float, float], model.consumer_domain[0])
    x = np.linspace(x_min, x_max, n_x, dtype=dtype)
    hx = cast(float, x[1] - x[0])

    y_min, y_max = cast(tuple[float, float], model.resource_domain[0])
    y = np.linspace(y_min, y_max, n_y, dtype=dtype)
    hy = cast(float, y[1] - y[0])

    t = np.linspace(0, T, n_t, dtype=dtype)
    delta_t = cast(float, t[1] - t[0])

    # 2. Obtenemos pesos de Simpson para las integrales
    weights_x = _get_simpson_weights(n_x, hx)
    weights_y = _get_simpson_weights(n_y, hy)

    # 3. Precalculamos parámetros del modelo que no dependen del tiempo
    lamb = model.mutation_rate * delta_t / (hx**2)
    A = get_sparse_matrix(lamb, n_x, border_type)

    kernel = model.resource_consumer_kernel(x[:, None], y[None, :]).astype(np.float64)
    consumer_growth_rate = model.consumer_growth_rate(x).astype(np.float64)
    consumer_decay = model.consumer_decay(x).astype(np.float64)
    resource_supply_rate = model.resource_supply_rate(y).astype(np.float64)
    resource_decay = model.resource_decay(y).astype(np.float64)

    initial_consumer = model.initial_consumer_distribution(x).astype(np.float64)

    # 4. Dependiendo del tipo de resource, usamos distintos métodos de solución
    # fmt: off
    if use_stationary_resource:
        consumer_distribution, resource_distribution = _solve_stationary_resource(
            n_t, n_x, n_y, delta_t, A, kernel, weights_x, weights_y,
            consumer_growth_rate, consumer_decay, resource_supply_rate, resource_decay,
            initial_consumer
        )
    else:
        initial_resource = model.initial_resource_distribution(y)
        consumer_distribution, resource_distribution = _solve_dynamic_resource(
            n_t, n_x, n_y, delta_t, A, kernel, weights_x, weights_y,
            consumer_growth_rate, consumer_decay, resource_supply_rate, resource_decay,
            initial_consumer, initial_resource
        )
    # fmt: on

    # 5. Obtenemos la cantidad total de consumidores a lo largo del tiempo
    consumer_quantity = compute_consumer_integral(
        consumer_distribution, np.ones_like(x), weights_x
    )

    return consumer_distribution, consumer_quantity, resource_distribution


def _solve_stationary_resource(
    n_t: int,
    n_x: int,
    n_y: int,
    delta_t: float,
    A: csc_matrix,
    weights_x: np.ndarray,
    weights_y: np.ndarray,
    kernel: np.ndarray,
    consumer_growth_rate: np.ndarray,
    consumer_decay: np.ndarray,
    resource_supply_rate: np.ndarray,
    resource_decay: np.ndarray,
    initial_consumer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Resuelve el sistema asumiendo un recurso que alcanza el equilibrio instantáneamente."""
    consumer_dist = np.zeros((n_t, n_x), dtype=dtype)
    resource_dist = np.zeros((n_t, n_y), dtype=dtype)

    consumer_dist[0, :] = initial_consumer

    # Inicialización del recurso estacionario en t=0
    resource_integral = compute_resource_integral(
        kernel, consumer_growth_rate, consumer_dist[0, :], weights_x
    )
    resource_dist[0, :] = resource_supply_rate / (resource_decay + resource_integral)

    for k in range(1, n_t):
        prev_consumer = consumer_dist[k - 1, :]
        prev_resource = resource_dist[k - 1, :]

        # Evolución del consumidor (Implícito)
        consumer_integral = compute_consumer_integral(kernel, prev_resource, weights_y)
        g = consumer_growth_rate * consumer_integral - consumer_decay
        A_implicit = A - delta_t * diags(g, format="csc")
        consumer_dist[k, :] = spsolve(A_implicit, prev_consumer)

        # Actualización algebraica del recurso estacionario
        resource_integral = compute_resource_integral(
            kernel, consumer_growth_rate, consumer_dist[k, :], weights_x
        )
        resource_dist[k, :] = resource_supply_rate / (
            resource_decay + resource_integral
        )

    return consumer_dist, resource_dist


def _solve_dynamic_resource(
    n_t: int,
    n_x: int,
    n_y: int,
    delta_t: float,
    A: csc_matrix,
    weights_x: np.ndarray,
    weights_y: np.ndarray,
    kernel: np.ndarray,
    consumer_growth_rate: np.ndarray,
    consumer_decay: np.ndarray,
    resource_supply_rate: np.ndarray,
    resource_decay: np.ndarray,
    initial_consumer: np.ndarray,
    initial_resource: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Resuelve el sistema acoplado dinámicamente en el tiempo (recurso dinámico)."""
    consumer_dist = np.zeros((n_t, n_x), dtype=dtype)
    resource_dist = np.zeros((n_t, n_y), dtype=dtype)

    consumer_dist[0, :] = initial_consumer
    resource_dist[0, :] = initial_resource

    for k in range(1, n_t):
        prev_consumer = consumer_dist[k - 1, :]
        prev_resource = resource_dist[k - 1, :]

        # Evolución del consumidor (Implícito)
        consumer_integral = compute_consumer_integral(kernel, prev_resource, weights_y)
        g = consumer_growth_rate * consumer_integral - consumer_decay
        A_implicit = A - delta_t * diags(g, format="csc")
        consumer_dist[k, :] = spsolve(A_implicit, prev_consumer)

        # Evolución del recurso (Explícito)
        resource_integral = compute_resource_integral(
            kernel, consumer_growth_rate, consumer_dist[k, :], weights_x
        )
        factor = delta_t * (resource_decay + resource_integral)
        resource_dist[k, :] = prev_resource * (1.0 - factor) + delta_t * resource_supply_rate

    return consumer_dist, resource_dist
