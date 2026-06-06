from typing import TYPE_CHECKING, cast, Literal
from dataclasses import dataclass

import numpy as np
from scipy.sparse import diags, eye, kron, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

from src.constants import dtype
from src.tools.grid import (
    TraitGrid,
    consumer_grid,
    resource_grid,
    time_grid,
)
from src.tools.model_components import (
    ModelCoefficients,
    build_model_coefficients,
    compute_consumer_integral,
    compute_resource_integral,
    compute_stationary_resource,
)

if TYPE_CHECKING:
    from src.model import Model


@dataclass(frozen=True)
class ThetaScheme:
    """
    Agrupa los parámetros y operadores matriciales asociados al esquema
    temporal θ utilizado para discretizar la ecuación del consumidor.

    Su propósito es centralizar todas las matrices precomputadas del
    esquema implícito, evitando reconstruirlas o pasarlas
    individualmente a cada función.

    Attributes
    ----------
    theta : float
        Parámetro del esquema θ.

    L : csc_matrix
        Discretización espacial del operador de mutación. Corresponde
        al laplaciano discreto multiplicado por la tasa de mutación.

    I_sparse : csc_matrix
        Matriz identidad de tamaño (n_x, n_x).

    B_theta : csc_matrix
        Parte explícita del esquema θ,

            B_theta = I - (1 - θ)ΔtL,

        que aparece en el lado derecho del sistema lineal para
        calcular n^{k+1}.
    """

    theta: float
    L: csc_matrix
    I_sparse: csc_matrix
    B_theta: csc_matrix


def build_theta_scheme(
    model: "Model",
    delta_t: float,
    border_type: Literal["neumann", "periodic"],
    theta: float,
) -> ThetaScheme:
    """Construye las matrices asociadas al esquema theta."""

    # L representa el término difusivo puro.
    # Multiplicamos por la tasa de mutación
    L = model.mutation_rate * build_laplacian(model.consumer_grid, border_type)

    I_sparse = eye(model.consumer_size, format="csc", dtype=dtype)

    B_theta = I_sparse - (1.0 - theta) * delta_t * L

    return ThetaScheme(
        theta=theta,
        L=L,
        I_sparse=I_sparse,
        B_theta=B_theta,
    )


def build_laplacian(
    grid: TraitGrid,
    border_type: Literal["neumann", "periodic"],
) -> csc_matrix:
    """
    Construye el laplaciano discreto multidimensional asociado
    a una malla cartesiana uniforme.

    Para cada dirección k se construye el operador unidimensional Δ_k
    discretizado mediante diferencias finitas de segundo orden
    sobre la malla correspondiente.

    El laplaciano multidimensional se obtiene mediante productos
    de Kronecker:

        Δ_h =
            Δ₁ ⊗ I₂ ⊗ ... ⊗ I_d
          + I₁ ⊗ Δ₂ ⊗ ... ⊗ I_d
          + ...
          + I₁ ⊗ ... ⊗ I_{d-1} ⊗ Δ_d.

    Esta representación es compatible con la vectorización de
    arreglos multidimensionales mediante ``ravel()`` y permite
    construir operadores en dimensión arbitraria reutilizando
    únicamente los laplacianos unidimensionales.

    Parameters
    ----------
    grid : TraitGrid
        Malla de rasgos sobre la cual se discretiza el operador.

    border_type : {"neumann", "periodic"}
        Tipo de condición de borde utilizada en cada dirección.

    Returns
    -------
    csc_matrix
        Matriz dispersa de forma

            (N_total, N_total),

        donde

            N_total = np.prod(grid.shape).

        Representa el laplaciano discreto positivo

            -Δ

        sobre la malla.
    """
    axis_laplacians = [
        _build_laplacian_1D(N, h, border_type)
        for N, h in zip(grid.shape, grid.spacing)
    ]

    identities = [
        eye(N, format="csc", dtype=dtype) for N in grid.shape
    ]

    # Inicializamos construcción de la matriz del laplaciano
    size = int(np.prod(grid.shape))
    L_total = csc_matrix((size, size), dtype=dtype)

    for axis, L_axis in enumerate(axis_laplacians):

        term = eye(1, format="csc", dtype=dtype)

        for j in range(len(grid.shape)):
            # Seleccionamos la matriz laplaciana 1D o una identidad, dependiendo de la coordenada
            factor = L_axis if j == axis else identities[j]

            term = kron(term, factor, format="csc")

        L_total += term

    return cast(csc_matrix, L_total)


def _build_laplacian_1D(
    N: int,
    h: float,
    border_type: Literal["neumann", "periodic"],
) -> csc_matrix:
    """Genera la matriz del operador -D * d^2/dx^2 (Laplaciano negativo).

    No incluye delta_t ni la identidad, permitiendo flexibilidad para el esquema theta.
    """
    inv_dx2 = 1.0 / (h**2)
    diag_central = 2.0 * inv_dx2 * np.ones(N)
    diag_lateral = -1.0 * inv_dx2 * np.ones(N - 1)

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


# Función orquestadora de diferencias finitas


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

    # Discretización del dominio
    x_grid = consumer_grid(model)
    y_grid = resource_grid(model)

    model.consumer_grid = x_grid
    model.resource_grid = y_grid

    x = x_grid.points
    y = y_grid.points

    _, dt = time_grid(model)

    # Obtenemos pesos de Simpson para las integrales
    weights_x = x_grid.simpson_weights
    weights_y = y_grid.simpson_weights

    # Precalculamos los operadores espaciales asociados a la mutación
    scheme = build_theta_scheme(model, dt, border_type, theta)

    # Precalculamos los parámetros funcionales del modelo
    coeffs = build_model_coefficients(model, x, y)

    # Inicialización de las soluciones
    consumer_dist = np.zeros((model.n_t, model.consumer_size), dtype=dtype)
    resource_dist = np.zeros((model.n_t, model.resource_size), dtype=dtype)

    # Evaluamos las condiciones iniciales
    consumer_dist[0, :] = model.initial_consumer_distribution(x).astype(dtype)
    resource_dist[0, :] = (
        compute_stationary_resource(consumer_dist[0, :], weights_x, coeffs)
        if use_stationary_resource
        else model.initial_resource_distribution(y).astype(dtype)
    )

    # Bucle temporal
    for k in range(1, model.n_t):
        prev_consumer = consumer_dist[k - 1, :]
        prev_resource = resource_dist[k - 1, :]

        consumer_dist[k, :] = _advance_consumer(
            prev_consumer, prev_resource, dt, weights_y, coeffs, scheme
        )
        resource_dist[k, :] = (
            compute_stationary_resource(consumer_dist[k, :], weights_x, coeffs)
            if use_stationary_resource
            else _advance_resource_explicit(
                prev_resource, consumer_dist[k, :], dt, weights_x, coeffs
            )
        )

    # Obtenemos cantidad de consumidores final
    consumer_quantity = consumer_dist @ weights_x

    return consumer_dist, consumer_quantity, resource_dist


def _advance_consumer(
    prev_consumer: np.ndarray,
    prev_resource: np.ndarray,
    delta_t: float,
    weights_y: np.ndarray,
    coeffs: ModelCoefficients,
    scheme: ThetaScheme,
) -> np.ndarray:
    """
    Resuelve un paso temporal de la ecuación del consumidor

        ∂ₜn = εΔn + g(x,R)n

    mediante un esquema θ:

        (I + θΔtL - ΔtGⁿ)nᵏ⁺¹ = (I - (1-θ)ΔtL)nᵏ.

    Aquí, L ≈ -εΔ es el operador de mutación discretizado y
    Gⁿ = diag(gⁿ), donde

        gⁿ(x)  = r(x) ∫ K(x,y)Rⁿ(y) dy - m₁(x).

    Devuelve la aproximación nᵏ⁺¹.
    """

    consumer_integral = compute_consumer_integral(
        coeffs.kernel, prev_resource, weights_y
    )
    growth = coeffs.consumer_growth_rate * consumer_integral - coeffs.consumer_decay

    A = (
        scheme.I_sparse
        + scheme.theta * delta_t * scheme.L
        - delta_t * diags(growth, format="csc", dtype=dtype)
    )

    rhs = scheme.B_theta @ prev_consumer

    return cast(np.ndarray, spsolve(A, rhs))


def _advance_resource_explicit(
    prev_resource: np.ndarray,
    consumer_distribution: np.ndarray,
    delta_t: float,
    weights_x: np.ndarray,
    coeffs: ModelCoefficients,
) -> np.ndarray:
    """
    Resuelve un paso temporal de la ecuación del recurso

        ∂ₜR = R_in - R(m₂ + J),

    donde

        J(y) = ∫ r(x)K(x,y)n(x)dx,

    mediante Euler explícito:

        Rᵏ⁺¹ = Rᵏ + Δt[R_in - Rᵏ(m₂ + Jᵏ)].

    Devuelve la aproximación Rᵏ⁺¹.
    """

    resource_integral = compute_resource_integral(
        coeffs.kernel,
        coeffs.consumer_growth_rate,
        consumer_distribution,
        weights_x,
    )

    factor = delta_t * (coeffs.resource_decay + resource_integral)

    return prev_resource * (1.0 - factor) + delta_t * coeffs.resource_supply_rate
