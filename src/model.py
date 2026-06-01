"""
Módulo que contiene la clase `model`, la cual contiene la API
general para resolver el modelo de dinámica adaptativa, en función
de los parámetros recibidos y el método de resolución seleccionado.

La función de esta clase es para orquestrar el pipeline de solución, las
funciones para resolver y gráficar deben estar en submódulos distintos.
"""

from typing import Callable, Literal

import numpy as np

from src.metodos.diferencias_finitas import solve_model_by_finite_differences
from src.metodos.espectral import solve_model_by_spectral
from src.plot import Plot

# Type para funciones que toman arreglos de numpy y retornan arreglos de numpy
VectorizedFunction = Callable[[np.ndarray], np.ndarray]


class Model:
    """
    Modelo de mutación y selección para poblaciones estructuradas por rasgos.

    La variable x representa el rasgo del consumidor,
    mientras que y representa el rasgo del recurso.

    El modelo evoluciona:
        n(x, t): distribución de consumidores
        R(y, t): distribución de recursos
    mediante un sistema de EDPs no locales con mutación.
    """

    def __init__(
        self,
        consumer_domain: np.ndarray,
        resource_domain: np.ndarray,
        mutation_rate: float,
    ) -> None:
        """
        Parameters
        ----------
        consumer_domain : np.ndarray
            Dominio discretizado de rasgos de consumidores (x).

            > Ejemplo: consumer_domain = [[0, 1], [-1, 1]] indica
            que x1 está en [0,1], y que x2 está en [-1,1].

        resource_domain : np.ndarray
            Dominio discretizado de rasgos de recursos (y).

            > Ejemplo: resource_domain = [[0, 1], [2, 3]] indica
            que y1 está en [0,1], y que y2 está en [2,3].

        mutation_rate : float
            Intensidad de mutación ε que multiplica
            el operador de difusión Δn.
        """

        assert len(consumer_domain.shape) == 2
        assert len(resource_domain.shape) == 2

        self.consumer_domain = consumer_domain
        self.resource_domain = resource_domain

        assert mutation_rate >= 0.0
        assert isinstance(mutation_rate, float)

        self.mutation_rate = mutation_rate

        # Aproximaciones numéricas de:
        # n(x, t): distribución de consumidores
        # R(y, t): distribución de recursos
        # Su construcción depende del método numérico utilizado para resolver la EDP.
        self.consumer_distribution: np.ndarray
        self.resource_distribution: np.ndarray

        # Clase para plotear las funciones
        self.plot = Plot(self)

    def set_consumer_growth_rate(self, vectorized_function: VectorizedFunction) -> None:
        """
        Define la tasa de crecimiento r(x)
        dependiente del rasgo del consumidor.

        Describe qué tan eficientemente se reproducen
        los consumidores con rasgo x.
        """
        self.consumer_growth_rate = vectorized_function

    def set_consumer_decay(self, vectorized_function: VectorizedFunction) -> None:
        """
        Define la mortalidad m1(x)
        de los consumidores.

        Determina la tasa de muerte natural
        de consumidores con rasgo x.
        """
        self.consumer_decay = vectorized_function

    def set_resource_supply_rate(self, vectorized_function: VectorizedFunction) -> None:
        """
        Define la tasa de suministro externo Rin(y)
        de recursos.

        Describe el flujo de entrada
        de recursos con rasgo y.
        """
        self.resource_supply_rate = vectorized_function

    def set_resource_decay(self, vectorized_function: VectorizedFunction) -> None:
        """
        Define la tasa de decaimiento m2(y)
        de los recursos.

        Determina la degradación o pérdida natural
        de recursos con rasgo y.
        """
        self.resource_decay = vectorized_function

    def set_resource_consumer_kernel(
        self, vectorized_function: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> None:
        """
        Define el kernel de interacción K(x, y).

        K(x, y) mide qué tan fuertemente un consumidor con
        rasgo x consume un recurso con rasgo y.
        """
        self.resource_consumer_kernel = vectorized_function

    def set_initial_data(
        self,
        initial_consumer_distribution: VectorizedFunction,
        initial_resource_distribution: VectorizedFunction,
    ) -> None:
        """
        Define las distribuciones iniciales
        de consumidores y recursos.

        Parameters
        ----------
        initial_consumer_distribution : callable
            Distribución inicial de consumidores n(x, 0).

        initial_resource_distribution : callable
            Distribución inicial de recursos R(y, 0).
        """

        self.initial_consumer_distribution = initial_consumer_distribution
        self.initial_resource_distribution = initial_resource_distribution

    def solve_by_finite_differences(
        self,
        T: float,
        n_t: int,
        n_x: int,
        n_y: int,
        border_type: Literal["neumann", "periodic"],
        theta: float,
        use_stationary_resource: bool,
    ) -> None:
        """
        Resuelve numéricamente el modelo de mutación-selección mediante
        diferencias finitas.

        El término difusivo asociado a las mutaciones se discretiza mediante
        diferencias finitas de segundo orden, mientras que los términos de
        selección e interacción consumidor-recurso se evalúan a través de
        cuadraturas numéricas. Dependiendo del valor de `theta` se obtienen
        distintos métodos de resolución:

        - `theta = 0`: Euler explícito.
        - `theta = 0.5`: Crank–Nicolson.
        - `theta = 1`: Euler implícito.

        Además, el recurso puede modelarse como dinámico o suponerse en
        equilibrio cuasi-estacionario en cada paso temporal.

        Parameters
        ----------
        T : float
            Tiempo final de simulación.

        n_t : int
            Número de puntos de discretización temporal.

        n_x : int
            Número de puntos de discretización del dominio de rasgos de los
            consumidores.

        n_y : int
            Número de puntos de discretización del dominio de rasgos de los
            recursos.

        border_type : {"neumann", "periodic"}
            Condición de borde utilizada para el operador de difusión.

            - `"neumann"`:
            Flujo nulo en los extremos del dominio
            (:math:`\\partial_x n = 0`).

            - `"periodic"`:
            Condiciones periódicas en la variable de rasgo.

        theta : float
            Parámetro del esquema temporal θ. Debe satisfacer `0 ≤ theta ≤ 1`.

            Valores habituales:

            - `0`: Euler explícito.
            - `0.5`: Crank–Nicolson.
            - `1`: Euler implícito.

        use_stationary_resource : bool
            - Si es ``True``, el recurso se considera estacionario y se obtiene explícitamente.
            - Si es ``False``, se resuelve la ecuación temporal del recurso.
        """
        assert 0 <= theta and theta <= 1
        assert T > 0
        assert isinstance(n_t, int) and n_t > 0
        assert isinstance(n_x, int) and n_x > 0
        assert isinstance(n_y, int) and n_y > 0
        assert border_type in ["neumann", "periodic"]
        assert isinstance(use_stationary_resource, bool)

        self.T = T
        self.n_t = n_t
        self.n_x = n_x
        self.n_y = n_y

        (
            self.consumer_distribution,
            self.consumer_quantity,
            self.resource_distribution,
        ) = solve_model_by_finite_differences(
            self, border_type, use_stationary_resource, theta
        )

    def solve_by_spectral(
        self,
        T: float,
        n_t: int,
        n_x: int,
        n_y: int,
        use_stationary_resource: bool = True,
    ) -> None:
        """Resuelve numéricamente el sistema mediante el método espectral."""
        assert T > 0
        assert isinstance(n_t, int) and n_t > 0
        assert isinstance(n_x, int) and n_x > 0
        assert isinstance(n_y, int) and n_y > 0
        assert isinstance(use_stationary_resource, bool)

        self.T = T
        self.n_t = n_t
        self.n_x = n_x
        self.n_y = n_y

        # Invocación al submódulo que acabamos de escribir
        (
            self.consumer_distribution,
            self.consumer_quantity,
            self.resource_distribution,
        ) = solve_model_by_spectral(self, use_stationary_resource)
