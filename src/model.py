
"""
Módulo que contiene la clase `model`, la cual contiene la API
general para resolver el modelo de dinámica adaptativa, en función
de los parámetros recibidos y el método de resolución seleccionado.

La función de esta clase es para orquestrar el pipeline de solución, las
funciones para resolver y gráficar deben estar en submódulos distintos.
"""


from typing import Callable

import numpy as np

# Type para funciones que toman arreglos de numpy y retornan arreglos de numpy
VectorizedFunction = Callable[[np.ndarray], np.ndarray]


class model:
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
        Parámetros
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

    def set_consumer_growth_rate(
        self, vectorized_function: VectorizedFunction
    ) -> None:
        """
        Define la tasa de crecimiento r(x)
        dependiente del rasgo del consumidor.

        Describe qué tan eficientemente se reproducen
        los consumidores con rasgo x.
        """
        self.consumer_growth_rate = vectorized_function

    def set_consumer_decay(
        self, vectorized_function: VectorizedFunction
    ) -> None:
        """
        Define la mortalidad m1(x)
        de los consumidores.

        Determina la tasa de muerte natural
        de consumidores con rasgo x.
        """
        self.consumer_decay = vectorized_function

    def set_resource_supply_rate(
        self, vectorized_function: VectorizedFunction
    ) -> None:
        """
        Define la tasa de suministro externo Rin(y)
        de recursos.

        Describe el flujo de entrada
        de recursos con rasgo y.
        """
        self.resource_supply_rate = vectorized_function

    def set_resource_decay(
        self, vectorized_function: VectorizedFunction
    ) -> None:
        """
        Define la tasa de decaimiento m2(y)
        de los recursos.

        Determina la degradación o pérdida natural
        de recursos con rasgo y.
        """
        self.resource_decay = vectorized_function

    def set_resource_consumer_kernel(
        self, vectorized_function: VectorizedFunction
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

        Parámetros
        ----------
        initial_consumer_distribution : callable
            Distribución inicial de consumidores n(x, 0).

        initial_resource_distribution : callable
            Distribución inicial de recursos R(y, 0).
        """

        self.initial_consumer_distribution = initial_consumer_distribution
        self.initial_resource_distribution = initial_resource_distribution
        
    def solve(self, method: str) -> None:
        """
        Resuelve numéricamente el sistema de EDPs
        de mutación-selección.

        Parámetros
        ----------
        method : str
            Método numérico utilizado para aproximar
            la solución.

            Ejemplos:
                - diferencias finitas
                - métodos espectrales
        """

        # self.consumer_distribution = ...
        # self.resource_distribution = ...
    