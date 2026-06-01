from typing import TYPE_CHECKING, cast, Literal

import numpy as np
from scipy.sparse import diags, csc_matrix, lil_matrix, eye
from scipy.sparse.linalg import spsolve

from src.funciones_generales import (
    dtype,
    get_simpson_weights,
    compute_consumer_integral,
    compute_resource_integral,
    compute_stationary_resource,
    consumer_grid,
    resource_grid,
    time_grid,
)

if TYPE_CHECKING:
    from src.model import Model


def get_laplacian_matrix(
    N_x: int,
    dx: float,
    border_type: Literal["neumann", "periodic"],
) -> csc_matrix:
    """Genera la matriz del operador -D * d^2/dx^2 (Laplaciano negativo).

    No incluye delta_t ni la identidad, permitiendo flexibilidad para el esquema theta.
    """
    inv_dx2 = 1.0 / (dx**2)
    diag_central = 2.0 * inv_dx2 * np.ones(N_x)
    diag_lateral = -1.0 * inv_dx2 * np.ones(N_x - 1)

    L = cast(
        lil_matrix,
        diags(
            [diag_lateral, diag_central, diag_lateral],  # type: ignore
            [-1, 0, 1],  # type: ignore
            format="lil",
            dtype=dtype,
        ).tolil(),
    )

    if border_type == "neumann":
        L[0, 1] = -2.0 * inv_dx2
        L[-1, -2] = -2.0 * inv_dx2
    elif border_type == "periodic":
        L[0, -1] = -1.0 * inv_dx2
        L[-1, 0] = -1.0 * inv_dx2
    else:
        raise ValueError(f"Tipo de borde desconocido: {border_type}")

    return cast(csc_matrix, L.tocsc())


def solve_model_by_finite_differences(
    model: "Model",
    border_type: Literal["neumann", "periodic"],
    use_stationary_resource: bool,
    theta: float = 0.5,  # Por defecto: Crank-Nicolson
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resuelve el sistema mutación-selección-recurso mediante
    diferencias finitas en el espacio y un esquema theta
    implícito para la ecuación del consumidor.

    Admite recursos estacionarios o dinámicos.
    """

    # 1. Discretización del dominio
    x, hx = consumer_grid(model)
    y, hy = resource_grid(model)
    _, delta_t = time_grid(model)

    # 2. Obtenemos pesos de Simpson para las integrales
    weights_x = get_simpson_weights(model.n_x, hx)
    weights_y = get_simpson_weights(model.n_y, hy)

    # 3. Operadores espaciales fijos de mutación (Laplaciano)
    # L representa el término difusivo puro. Multiplicamos por la tasa de mutación.
    L = model.mutation_rate * get_laplacian_matrix(model.n_x, hx, border_type)
    I_sparse = eye(model.n_x, format="csc", dtype=dtype)

    # Precalcular la matriz explícita del pasado (B) que no cambia en el bucle
    B_theta = I_sparse - (1.0 - theta) * delta_t * L

    # 4. Parámetros funcionales del modelo en float64
    kernel = model.resource_consumer_kernel(x[:, None], y[None, :]).astype(dtype)
    consumer_growth_rate = model.consumer_growth_rate(x).astype(dtype)
    consumer_decay = model.consumer_decay(x).astype(dtype)
    resource_supply_rate = model.resource_supply_rate(y).astype(dtype)
    resource_decay = model.resource_decay(y).astype(dtype)

    initial_consumer = model.initial_consumer_distribution(x).astype(dtype)

    # 5. Bifurcación del solucionador según la naturaleza del recurso
    # fmt: off
    if use_stationary_resource:
        consumer_distribution, resource_distribution = _solve_stationary_resource(
            model.n_t, model.n_x, model.n_y, delta_t, theta, I_sparse, L, B_theta, weights_x, weights_y, kernel,
            consumer_growth_rate, consumer_decay, resource_supply_rate, resource_decay,
            initial_consumer
        )
    else:
        initial_resource = model.initial_resource_distribution(y).astype(dtype)
        consumer_distribution, resource_distribution = _solve_dynamic_resource(
            model.n_t, model.n_x, model.n_y, delta_t, theta, I_sparse, L, B_theta, weights_x, weights_y, kernel,
            consumer_growth_rate, consumer_decay, resource_supply_rate, resource_decay,
            initial_consumer, initial_resource
        )
    # fmt: on

    # 6. Cantidad global final
    consumer_quantity = compute_consumer_integral(
        consumer_distribution, np.ones_like(x, dtype=dtype), weights_x
    )

    return consumer_distribution, consumer_quantity, resource_distribution


def _solve_stationary_resource(
    n_t: int,
    n_x: int,
    n_y: int,
    delta_t: float,
    theta: float,
    I_sparse: csc_matrix,
    L: csc_matrix,
    B_theta: csc_matrix,
    weights_x: np.ndarray,
    weights_y: np.ndarray,
    kernel: np.ndarray,
    consumer_growth_rate: np.ndarray,
    consumer_decay: np.ndarray,
    resource_supply_rate: np.ndarray,
    resource_decay: np.ndarray,
    initial_consumer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solución con recurso estacionario utilizando el esquema theta general."""
    consumer_dist = np.zeros((n_t, n_x), dtype=dtype)
    resource_dist = np.zeros((n_t, n_y), dtype=dtype)

    consumer_dist[0, :] = initial_consumer

    # Inicialización del recurso en t=0
    resource_dist[0, :] = compute_stationary_resource(
        kernel,
        consumer_growth_rate,
        consumer_dist[0, :],
        weights_x,
        resource_supply_rate,
        resource_decay,
    )

    for k in range(1, n_t):
        prev_consumer = consumer_dist[k - 1, :]
        prev_resource = resource_dist[k - 1, :]

        # 1. Integrales y tasa de crecimiento basada en el paso anterior
        consumer_integral = compute_consumer_integral(kernel, prev_resource, weights_y)
        g = consumer_growth_rate * consumer_integral - consumer_decay

        # 2. Ensamblar sistema Theta: (I + theta*dt*L - dt*G) u^{n+1} = B_theta u^n
        A_implicit = (
            I_sparse
            + theta * delta_t * L
            - delta_t * diags(g, format="csc", dtype=dtype)
        )
        lado_derecho = B_theta @ prev_consumer

        consumer_dist[k, :] = spsolve(A_implicit, lado_derecho)

        # 3. Actualización algebraica del recurso de forma síncrona
        resource_dist[k, :] = compute_stationary_resource(
            kernel,
            consumer_growth_rate,
            consumer_dist[k, :],
            weights_x,
            resource_supply_rate,
            resource_decay,
        )

    return consumer_dist, resource_dist


def _solve_dynamic_resource(
    n_t: int,
    n_x: int,
    n_y: int,
    delta_t: float,
    theta: float,
    I_sparse: csc_matrix,
    L: csc_matrix,
    B_theta: csc_matrix,
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
    """Solución con recurso acoplado dinámico utilizando el esquema theta general."""
    consumer_dist = np.zeros((n_t, n_x), dtype=dtype)
    resource_dist = np.zeros((n_t, n_y), dtype=dtype)

    consumer_dist[0, :] = initial_consumer
    resource_dist[0, :] = initial_resource

    for k in range(1, n_t):
        prev_consumer = consumer_dist[k - 1, :]
        prev_resource = resource_dist[k - 1, :]

        # 1. Evolución del consumidor vía esquema Theta
        consumer_integral = compute_consumer_integral(kernel, prev_resource, weights_y)
        g = consumer_growth_rate * consumer_integral - consumer_decay

        A_implicit = (
            I_sparse
            + theta * delta_t * L
            - delta_t * diags(g, format="csc", dtype=dtype)
        )
        lado_derecho = B_theta @ prev_consumer

        consumer_dist[k, :] = spsolve(A_implicit, lado_derecho)

        # 2. Evolución explícita del recurso (con el consumidor ya actualizado en k)
        resource_integral = compute_resource_integral(
            kernel, consumer_growth_rate, consumer_dist[k, :], weights_x
        )
        factor = delta_t * (resource_decay + resource_integral)
        resource_dist[k, :] = (
            prev_resource * (1.0 - factor) + delta_t * resource_supply_rate
        )

    return consumer_dist, resource_dist
