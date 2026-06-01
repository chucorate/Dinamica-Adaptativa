from typing import TYPE_CHECKING, cast

import numpy as np

from src.funciones_generales import dtype, get_simpson_weights

if TYPE_CHECKING:
    from src.model import Model


def _compute_growth_rate(
    kernel: np.ndarray,
    resource: np.ndarray,
    weights_y: np.ndarray,
    consumer_growth_rate: np.ndarray,
    consumer_decay: np.ndarray,
) -> np.ndarray:
    """Calcula la tasa de selección g(x) = r(x) * ∫ K(x, y) R(y) dy - m1(x)"""
    weighted_resource = weights_y * resource
    consumer_integral = kernel @ weighted_resource
    return consumer_growth_rate * consumer_integral - consumer_decay


def solve_model_by_spectral(
    model: "Model",
    T: float,
    n_t: int,
    n_x: int,
    n_y: int,
    use_stationary_resource: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resuelve el sistema de mutación-selección utilizando el método pseudo-espectral de Fourier
    para el espacio y un esquema integrador temporal IMEX de segundo orden.
    """
    if not use_stationary_resource:
        raise NotImplementedError(
            "Por el momento solo se admitre recurso estacionario en el método espectral."
        )

    # 1. Discretización del dominio espacial
    x_min, x_max = cast(tuple[float, float], model.consumer_domain[0])
    x = np.linspace(x_min, x_max, n_x, dtype=dtype)
    hx = cast(float, x[1] - x[0])

    y_min, y_max = cast(tuple[float, float], model.resource_domain[0])
    y = np.linspace(y_min, y_max, n_y, dtype=dtype)
    hy = cast(float, y[1] - y[0])

    # Discretización temporal
    t = np.linspace(0, T, n_t, dtype=dtype)
    delta_t = cast(float, t[1] - t[0])

    # Pesos de Simpson para las integrales no locales
    weights_x = get_simpson_weights(n_x, hx)
    weights_y = get_simpson_weights(n_y, hy)

    # 2. Configuración de frecuencias de Fourier para el Laplaciano (x)
    frequencies = np.fft.fftfreq(n_x, d=hx)
    k = 2.0 * np.pi * frequencies
    laplacian_kernel = -model.mutation_rate * (k**2)

    # Factor implícito de difusión (Vector de tamaño n_x)
    implicit_diffusion_factor = 1.0 / (1.0 - delta_t * laplacian_kernel)

    # 3. Parámetros del modelo y normalización forzada del Kernel
    kernel = model.resource_consumer_kernel(x[:, None], y[None, :]).astype(dtype)
    consumer_growth_rate = model.consumer_growth_rate(x).astype(dtype)
    consumer_decay = model.consumer_decay(x).astype(dtype)
    resource_supply_rate = model.resource_supply_rate(y).astype(dtype)
    resource_decay = model.resource_decay(y).astype(dtype)

    # 4. Inicialización de estructuras de almacenamiento matrices (n_t, n_x) y (n_t, n_y)
    consumer_dist = np.zeros((n_t, n_x), dtype=dtype)
    resource_dist = np.zeros((n_t, n_y), dtype=dtype)

    # Condición inicial
    consumer_dist[0, :] = model.initial_consumer_distribution(x).astype(dtype)

    # Estado cuasi-estático inicial del recurso en t=0
    weighted_consumer = weights_x * consumer_growth_rate * consumer_dist[0, :]
    resource_integral = kernel.T @ weighted_consumer
    resource_dist[0, :] = resource_supply_rate / (resource_decay + resource_integral)

    # Bucle temporal pseudo-espectral corregido
    for n in range(1, n_t):
        prev_consumer = consumer_dist[n - 1, :].copy()
        prev_resource = resource_dist[n - 1, :].copy()

        # --- PASO DE PREDICCIÓN (IMEX Euler) ---
        g_n = _compute_growth_rate(
            kernel, prev_resource, weights_y, consumer_growth_rate, consumer_decay
        )
        reaccion_real = g_n * prev_consumer

        # Transformadas de Fourier (vectores de tamaño n_x)
        consumer_hat = np.fft.fft(prev_consumer)
        reaccion_hat = np.fft.fft(reaccion_real)

        # Avanzar el predictor en Fourier y regresar al espacio real
        pred_consumer_hat = (
            consumer_hat + delta_t * reaccion_hat
        ) * implicit_diffusion_factor
        pred_consumer = np.real(np.fft.ifft(pred_consumer_hat))

        # Corrección de umbral para evitar que se reduzca a un escalar
        pred_consumer = np.clip(pred_consumer, 0.0, None)

        # Evaluar el recurso predicho futuro síncrono
        weighted_pred_consumer = weights_x * consumer_growth_rate * pred_consumer
        pred_resource_integral = kernel.T @ weighted_pred_consumer
        pred_resource = resource_supply_rate / (resource_decay + pred_resource_integral)

        # --- PASO DE CORRECCIÓN (IMEX Crank-Nicolson / Heun Espectral) ---
        g_pred = _compute_growth_rate(
            kernel, pred_resource, weights_y, consumer_growth_rate, consumer_decay
        )
        reaccion_pred_real = g_pred * pred_consumer

        # Promediar crecimiento en Fourier
        reaccion_promedio_hat = 0.5 * (reaccion_hat + np.fft.fft(reaccion_pred_real))

        # Inversa definitiva para el paso n
        final_consumer_hat = (
            consumer_hat + delta_t * reaccion_promedio_hat
        ) * implicit_diffusion_factor

        # Guardar estrictamente en la fila correspondiente de la matriz
        consumer_dist[n, :] = np.real(np.fft.ifft(final_consumer_hat))
        consumer_dist[n, :] = np.clip(consumer_dist[n, :], 0.0, None)

        # 4. Actualización definitiva del recurso estacionario en la matriz
        weighted_final_consumer = weights_x * consumer_growth_rate * consumer_dist[n, :]
        final_resource_integral = kernel.T @ weighted_final_consumer
        resource_dist[n, :] = resource_supply_rate / (
            resource_decay + final_resource_integral
        )

    return consumer_dist, resource_dist
