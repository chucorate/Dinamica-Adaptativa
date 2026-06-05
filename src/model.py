"""
Módulo que contiene la clase `model`, la cual contiene la API
general para resolver el modelo de dinámica adaptativa, en función
de los parámetros recibidos y el método de resolución seleccionado.

La función de esta clase es para orquestrar el pipeline de solución, las
funciones para resolver y gráficar deben estar en submódulos distintos.
"""

from typing import Callable, Literal

import numpy as np

from src.tools.grid import TraitGrid
from src.metodos.diferencias_finitas import solve_model_by_finite_differences
from src.plot import Plot

# Type para funciones que toman arreglos de numpy y retornan arreglos de numpy
# recibe (N,d) y retorna (N,)
TraitFunction = Callable[[np.ndarray], np.ndarray]
KernelFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _validate_trait_function(f: TraitFunction, dimension: int) -> None:
    test = np.zeros((5, dimension))
    out = f(test)

    if not isinstance(out, np.ndarray):
        raise TypeError("La función debe retornar un np.ndarray.")

    if out.shape != (5,):
        raise ValueError(f"Los input y output de la función debe tener el mismo shape.")


def _validate_kernel_function(
    kernel: KernelFunction, consumer_dimension: int, resource_dimension: int
) -> None:
    x = np.zeros((7, consumer_dimension))
    y = np.zeros((11, resource_dimension))

    K = kernel(x, y)

    if not isinstance(K, np.ndarray):
        raise TypeError("El kernel debe retornar un np.ndarray.")

    if K.shape != (7, 11):
        raise ValueError(f"Shape inválido: {K.shape}. Se esperaba (7,11).")


def _sanitize_discretion_points(
    discretion_points: int | tuple[int, ...],
    domain: np.ndarray,
) -> tuple[int, ...]:
    """
    Verifica que los puntos de discretización dados sean válidos.

    Retorna una tupla con los valores de discretización para cada componente del rasgo.
    En caso de que se haya entregado un entero, se pasa a tupla con un solo parámetro.
    """
    if isinstance(discretion_points, int):
        assert discretion_points > 1
        discretion_points = (discretion_points,)

    elif isinstance(discretion_points, tuple):
        for n in discretion_points:
            assert n > 1 and isinstance(n, int)

    else:
        raise ValueError(f"{discretion_points} no es entero ni tupla de enteros.")

    if len(discretion_points) != len(domain):
        raise ValueError(
            "Los puntos de discretización y el dominio tienen dimensiones incompatibles:\n"
            f"Puntos de discretización: {discretion_points}, tamaño {len(discretion_points)}\n"
            f"Dominio: {domain}, tamaño {len(domain)}"
        )

    return discretion_points


class Model:
    """
    Modelo de mutación y selección para poblaciones estructuradas por rasgos.

    La variable x representa el rasgo del consumidor,
    mientras que y representa el rasgo del recurso.

    El modelo evoluciona:
        n(x, t): distribución de consumidores
        R(y, t): distribución de recursos
    mediante un sistema de EDPs no locales con mutación.

    Attributes
    ----------
    consumer_distribution : np.ndarray
        Solución discreta del consumidor.

        Tiene forma

            (n_t, N_consumer)

        donde

            N_consumer = np.prod(n_x).

    resource_distribution : np.ndarray
        Solución discreta del recurso.

        Tiene forma

            (n_t, N_resource)

        donde

            N_resource = np.prod(n_y).

    consumer_quantity : np.ndarray
        Cantidad total de consumidores en cada instante temporal.

        Tiene forma

            (n_t,).

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
        # rho(t): cantidad de consumidores
        # Su construcción depende del método numérico utilizado para resolver la EDP.
        self.consumer_distribution: np.ndarray
        self.resource_distribution: np.ndarray
        self.consumer_quantity: np.ndarray

        # Clase para plotear las funciones
        self.plot = Plot(self)

        # Atributos auxiliares
        self.n_x: tuple[int, ...]
        self.n_y: tuple[int, ...]
        self.consumer_grid: TraitGrid
        self.resource_grid: TraitGrid

    @property
    def consumer_dimension(self) -> int:
        """
        Dimensión del espacio de rasgos de los consumidores.

        Equivale al número de componentes del vector de rasgos x.
        """
        return len(self.consumer_domain)

    @property
    def resource_dimension(self) -> int:
        """
        Dimensión del espacio de rasgos de los recursos.

        Equivale al número de componentes del vector de rasgos y.
        """
        return len(self.resource_domain)

    @property
    def consumer_size(self) -> int:
        """
        Número total de puntos de la malla de consumidores.

        Si

            n_x = (N₁,...,N_d),

        entonces

            consumer_size = N₁⋯N_d.

        Coincide con

            consumer_grid.points.shape[0].

        """
        return int(np.prod(self.n_x))

    @property
    def resource_size(self) -> int:
        """
        Número total de puntos de la malla de recursos.

        Si

            n_y = (M₁,...,M_d),

        entonces

            resource_size = M₁⋯M_d.

        Coincide con

            resource_grid.points.shape[0].
        """
        return int(np.prod(self.n_y))

    def set_consumer_growth_rate(self, vectorized_function: TraitFunction) -> None:
        """
        Define la tasa de crecimiento r(x) dependiente del rasgo del consumidor.

        Describe qué tan eficientemente se reproducen los consumidores con rasgo x.

        La función debe recibir una matriz

            x.shape = (N, d_x)

        donde cada fila representa un punto del espacio
        de rasgos de consumidores,

            x[i] = (x₁,...,x_d),

        y retornar un vector

            r.shape = (N,)

        tal que

            r[i] = r(x[i]).

        """
        _validate_trait_function(vectorized_function, self.consumer_dimension)
        self.consumer_growth_rate = vectorized_function

    def set_consumer_decay(self, vectorized_function: TraitFunction) -> None:
        """
        Define la mortalidad m1(x) de los consumidores.

        Determina la tasa de muerte natural de consumidores con rasgo x.

        La función debe recibir

            x.shape = (N, d_x)

        y retornar

            m.shape = (N,)

        donde

            m[i] = m₁(x[i]).

        """
        _validate_trait_function(vectorized_function, self.consumer_dimension)
        self.consumer_decay = vectorized_function

    def set_resource_supply_rate(self, vectorized_function: TraitFunction) -> None:
        """
        Define la tasa de suministro externo Rin(y) de recursos.

        Describe el flujo de entrada de recursos con rasgo y.

        La función debe recibir

            y.shape = (N, d_y)

        y retornar

            Rin.shape = (N,)

        donde

            Rin[i] = Rin(y[i]).

        """
        _validate_trait_function(vectorized_function, self.resource_dimension)
        self.resource_supply_rate = vectorized_function

    def set_resource_decay(self, vectorized_function: TraitFunction) -> None:
        """
        Define la tasa de decaimiento m2(y) de los recursos.

        Determina la degradación o pérdida natural de recursos con rasgo y.

        La función debe recibir

            y.shape = (N, d_y)

        y retornar

            m2.shape = (N,)

        donde

            m2[i] = m₂(y[i]).

        """
        _validate_trait_function(vectorized_function, self.resource_dimension)
        self.resource_decay = vectorized_function

    def set_resource_consumer_kernel(self, kernel_function: KernelFunction) -> None:
        """
        Define el kernel de interacción K(x, y).

        K(x, y) mide qué tan fuertemente un consumidor con
        rasgo x consume un recurso con rasgo y.

        La función debe recibir

            x.shape = (N_consumer, d_x)
            y.shape = (N_resource, d_y)

        y retornar

            K.shape = (N_consumer, N_resource)

        donde

            K[i,j] = K(x_i,y_j).

        """
        _validate_kernel_function(
            kernel_function, self.consumer_dimension, self.resource_dimension
        )
        self.resource_consumer_kernel = kernel_function

    def set_initial_data(
        self,
        initial_consumer_distribution: TraitFunction,
        initial_resource_distribution: TraitFunction,
    ) -> None:
        """
        Define las distribuciones iniciales de consumidores y recursos.

        Las funciones deben recibir las coordenadas de la malla
        de rasgos y retornar los valores iniciales evaluados
        en cada punto.

        Parameters
        ----------
        initial_consumer_distribution : TraitFunction
            Distribución inicial de consumidores n(x, 0).

            Debe recibir

                x.shape = (N_consumer, d_x)

            y retornar

                n0.shape = (N_consumer,)

            donde

                n0[i] = n(0, x[i]).

        initial_resource_distribution : TraitFunction
            Distribución inicial de recursos R(y, 0).

            Debe recibir

                y.shape = (N_resource, d_y)

            y retornar

                R0.shape = (N_resource,)

            donde

                R0[i] = R(0, y[i]).

        """
        _validate_trait_function(initial_consumer_distribution, self.consumer_dimension)
        _validate_trait_function(initial_resource_distribution, self.resource_dimension)
        self.initial_consumer_distribution = initial_consumer_distribution
        self.initial_resource_distribution = initial_resource_distribution

    def solve_by_finite_differences(
        self,
        T: float,
        n_t: int,
        n_x: int | tuple[int, ...],
        n_y: int | tuple[int, ...],
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

        n_x : int | tuple[int, ...]
            Resolución espacial utilizada para discretizar cada dimensión
            del espacio de rasgos de los consumidores.

            Si se entrega un entero n_x, se interpreta como (n_x,)

        n_y : int | tuple[int, ...]
            Resolución espacial utilizada para discretizar cada dimensión
            del espacio de rasgos de los recursos.

            Si se entrega un entero n_y, se interpreta como (n_y,)

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
            - Si es `True`, el recurso se considera estacionario y se obtiene explícitamente.
            - Si es `False`, se resuelve la ecuación temporal del recurso.
        """
        # Checkear que los parámetros del método sean válidos
        assert 0 <= theta and theta <= 1
        assert border_type in ["neumann", "periodic"]
        assert isinstance(use_stationary_resource, bool)

        # Checkear parámetros asociados al tiempo
        assert T > 0
        assert isinstance(n_t, int) and n_t > 0
        self.T = T
        self.n_t = n_t

        # Checkear parámetros asociados al espacio
        self.n_x = _sanitize_discretion_points(n_x, self.consumer_domain)
        self.n_y = _sanitize_discretion_points(n_y, self.resource_domain)

        (
            self.consumer_distribution,
            self.consumer_quantity,
            self.resource_distribution,
        ) = solve_model_by_finite_differences(
            self, border_type, use_stationary_resource, theta
        )
