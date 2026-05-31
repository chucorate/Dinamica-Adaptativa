"""
Submódulo para graficar las soluciones otorgadas por `model`.
"""

from typing import TYPE_CHECKING, Literal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animate

if TYPE_CHECKING:
    from src.model import Model


def _consumer_grid(model) -> np.ndarray:
    xmin, xmax = model.consumer_domain[0]
    nx = model.consumer_distribution.shape[1]

    return np.linspace(xmin, xmax, nx)


def _resource_grid(model) -> np.ndarray:
    ymin, ymax = model.resource_domain[0]
    ny = model.resource_distribution.shape[1]

    return np.linspace(ymin, ymax, ny)


def _time_grid(model) -> np.ndarray:
    nt = model.consumer_distribution.shape[0]

    if hasattr(model, "T"):
        return np.linspace(0.0, model.T, nt)

    return np.arange(nt)


def plot_solution(
    model: "Model",
    solution: Literal["consumer-density", "consumer-quantity", "resource"],
    plot_type: Literal["heatmap", "contour", "surface"] = "heatmap",
    **kwargs,
) -> None:
    # solucion de la forma u(t,y)
    # en eje x va el rasgo, en eje y va el tiempo (de abajo hacia arriba)
    """
    Parameters
    ----------
    solution :
        - consumer-density
        - consumer-quantity
        - resource

    plot_type :
        - heatmap
        - contour
        - surface

    kwargs :
        Se pasan directamente al backend
        de matplotlib correspondiente.
    """

    t = _time_grid(model)

    # POBLACIÓN TOTAL
    if solution == "consumer-quantity":
        plt.plot(t, model.consumer_quantity, **kwargs)
        plt.xlabel("Time")
        plt.ylabel("Consumer quantity")

        plt.show()
        return

    # CONSUMIDORES
    if solution == "consumer-density":
        x = _consumer_grid(model)
        Z = model.consumer_distribution
        xlabel = "Consumer trait"

    # RECURSOS
    elif solution == "resource":
        x = _resource_grid(model)
        Z = model.resource_distribution
        xlabel = "Resource trait"

    else:
        raise ValueError(
            f"Unknown solution '{solution}'"
        )

    X, T = np.meshgrid(x, t)

    # Seleccionar tipo de gráfico

    # HEATMAP
    if plot_type == "heatmap":
        default_kwargs = dict(
            aspect="auto",
            origin="lower",
            extent=[x.min(), x.max(), t.min(), t.max()],
        )

        default_kwargs.update(kwargs)

        plt.imshow( Z, **default_kwargs)
        plt.xlabel(xlabel)
        plt.ylabel("Time")
        plt.colorbar()

    # CONTOUR
    elif plot_type == "contour":
        plt.contourf(X, T, Z, **kwargs)
        plt.xlabel(xlabel)
        plt.ylabel("Time")
        plt.colorbar()

    # SURFACE
    elif plot_type == "surface":

        fig = plt.figure()

        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, T, Z, **kwargs)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Time")
        ax.set_zlabel("Density")

    else:
        raise ValueError(f"Unknown plot_type '{plot_type}'")
    
    plt.show()


def plot_kernel(model: "Model", format: Literal["heat", "3D"], **kwargs) -> None:
    # Grafico 3D o de calor, para plotear kernels
    return


def plot_1D(
    model: "Model",
    function: Literal["c-growth, c-decay, r-growth, r-decay"],
    **kwargs,
) -> None:
    return
