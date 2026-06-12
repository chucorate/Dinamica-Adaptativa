# WIP, no funciona del todo bien


from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np

from src.funciones_generales import (
    dtype,
    build_model_coefficients,
    get_simpson_weights,
    compute_consumer_integral,
    compute_stationary_resource,
    consumer_grid,
    resource_grid,
    time_grid,
)

if TYPE_CHECKING:
    from src.model import Model


@dataclass(frozen=True)
class SpectralScheme:
    """
    Agrupa todos los operadores precomputados asociados
    al método pseudo-espectral de Fourier.

    Attributes
    ----------
    implicit_diffusion_factor : np.ndarray
        Factor diagonal utilizado para tratar implícitamente
        la difusión en el espacio de Fourier.

    laplacian_kernel : np.ndarray
        Símbolo espectral del operador de mutación.
    """

    implicit_diffusion_factor: np.ndarray
    laplacian_kernel: np.ndarray


def build_spectral_scheme(
    model: "Model",
    hx: float,
    delta_t: float,
) -> SpectralScheme:
    """Construye los operadores espectrales precomputados."""

    frequencies = np.fft.fftfreq(model.n_x, d=hx)
    k = 2.0 * np.pi * frequencies

    laplacian_kernel = -model.mutation_rate * (k**2)

    implicit_diffusion_factor = 1.0 / (1.0 - delta_t * laplacian_kernel)

    return SpectralScheme(
        implicit_diffusion_factor=implicit_diffusion_factor,
        laplacian_kernel=laplacian_kernel,
    )


# Función orquestadora de método espectral


def solve_model_by_spectral(
    model: "Model",
    use_stationary_resource: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resuelve el sistema de mutación-selección utilizando el método pseudo-espectral de Fourier
    para el espacio y un esquema integrador temporal IMEX de segundo orden.
    """
    if not use_stationary_resource:
        raise NotImplementedError(
            "Por el momento solo se admitre recurso estacionario en el método espectral."
        )

    # 1. Discretización del dominio espacial
    x, hx = consumer_grid(model)
    y, hy = resource_grid(model)
    _, dt = time_grid(model)

    # Pesos de Simpson para las integrales no locales
    weights_x = get_simpson_weights(model.n_x, hx)
    weights_y = get_simpson_weights(model.n_y, hy)

    coeffs = build_model_coefficients(model, x, y)
    scheme = build_spectral_scheme(model, hx, dt)

    # Inicialización de las soluciones
    consumer_dist = np.zeros((model.n_t, model.n_x), dtype=dtype)
    resource_dist = np.zeros((model.n_t, model.n_y), dtype=dtype)

    # Evaluamos las condiciones iniciales
    consumer_dist[0, :] = model.initial_consumer_distribution(x).astype(dtype)
    resource_dist[0, :] = (
        compute_stationary_resource(consumer_dist[0, :], weights_x, coeffs)
        if use_stationary_resource
        else model.initial_resource_distribution(y).astype(dtype)
    )

    # Bucle temporal pseudo-espectral corregido
    for k in range(1, model.n_t):
        prev_consumer = consumer_dist[k - 1, :]
        prev_resource = resource_dist[k - 1, :]

        # Crecimiento evaluado en el estado actual
        g_prev = (
            coeffs.consumer_growth_rate
            * compute_consumer_integral(coeffs.kernel, prev_resource, weights_y)
            - coeffs.consumer_decay
        )
        # Predictor IMEX-Euler
        pred_consumer = spectral_predictor(prev_consumer, g_prev, dt, scheme)
        pred_resource = compute_stationary_resource(pred_consumer, weights_x, coeffs)

        # Crecimiento evaluado en el predictor
        g_pred = (
            coeffs.consumer_growth_rate
            * compute_consumer_integral(coeffs.kernel, pred_resource, weights_y)
            - coeffs.consumer_decay
        )
        # Corrector IMEX-Heun
        consumer_dist[k, :] = spectral_corrector(
            prev_consumer, pred_consumer, g_prev, g_pred, dt, scheme
        )
        resource_dist[k, :] = compute_stationary_resource(
            consumer_dist[k, :], weights_x, coeffs
        )

    consumer_quantity = compute_consumer_integral(
        consumer_dist, np.ones_like(x, dtype=dtype), weights_x
    )

    return consumer_dist, consumer_quantity, resource_dist


def spectral_predictor(
    prev_consumer: np.ndarray,
    growth: np.ndarray,
    delta_t: float,
    scheme: SpectralScheme,
) -> np.ndarray:
    """
    Realiza el paso predictor IMEX-Euler

        n* = (I - ΔtD)^(-1) [nᵏ + Δt gᵏ nᵏ].

    """

    reaction = growth * prev_consumer

    consumer_hat = np.fft.fft(prev_consumer)
    reaction_hat = np.fft.fft(reaction)

    predicted_hat = (
        consumer_hat + delta_t * reaction_hat
    ) * scheme.implicit_diffusion_factor

    predicted = np.real(np.fft.ifft(predicted_hat))

    return np.clip(predicted, 0.0, None)


def spectral_corrector(
    prev_consumer: np.ndarray,
    predicted_consumer: np.ndarray,
    growth_prev: np.ndarray,
    growth_pred: np.ndarray,
    delta_t: float,
    scheme: SpectralScheme,
) -> np.ndarray:
    """
    Realiza el paso corrector IMEX-Heun

        nᵏ⁺¹ = (I - ΔtD)^(-1) [nᵏ + Δt/2(Fᵏ + F*)].

    """

    reaction_prev = growth_prev * prev_consumer
    reaction_pred = growth_pred * predicted_consumer

    consumer_hat = np.fft.fft(prev_consumer)

    reaction_avg_hat = 0.5 * (np.fft.fft(reaction_prev) + np.fft.fft(reaction_pred))

    final_hat = (
        consumer_hat + delta_t * reaction_avg_hat
    ) * scheme.implicit_diffusion_factor

    final_consumer = np.real(np.fft.ifft(final_hat))

    return np.clip(final_consumer, 0.0, None)
