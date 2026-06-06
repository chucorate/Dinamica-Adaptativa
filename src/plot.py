"""
Submódulo para graficar las soluciones otorgadas por `model`.
"""

from typing import TYPE_CHECKING, Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from IPython.display import HTML
from matplotlib import rc
from matplotlib import rcParams

rc("animation", html="jshtml")
rcParams["animation.embed_limit"] = 200
if TYPE_CHECKING:
    from src.model import Model


def _require_dimension(model: "Model", dimensions: int) -> None:
    assert isinstance(dimensions, int) and dimensions >= 1

    if model.consumer_dimension != dimensions:
        raise ValueError(
            f"Esta visualización requiere d_x = {dimensions}.\n"
            f"Dimensión de consumidores actual = {model.consumer_dimension}."
        )

    if model.resource_dimension != dimensions:
        raise ValueError(
            f"Esta visualización requiere d_y = {dimensions}.\n"
            f"Dimensión de recursos actual = {model.resource_dimension}."
        )


def plot_solution_1D(
    model: "Model",
    plot_type: Literal["heatmap", "contour"],
    **kwargs,
) -> None:
    _require_dimension(model, dimensions=1)

    t = model.time_grid
    x = model.consumer_grid.points[:, 0]
    y = model.resource_grid.points[:, 0]

    consumer = (
        x,
        model.consumer_distribution,
        "Consumer trait",
        "Consumer density",
    )
    resource = (
        y,
        model.resource_distribution,
        "Resource trait",
        "Resource density",
    )
    datasets = (consumer, resource)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    for ax, (x, Z, xlabel, title) in zip(axes, datasets):
        if plot_type == "heatmap":
            artist = ax.imshow(
                Z,
                aspect="auto",
                origin="lower",
                extent=[x.min(), x.max(), t.min(), t.max()],
                **kwargs,
            )
        else:
            X, T = np.meshgrid(x, t)
            artist = ax.contourf(X, T, Z, **kwargs)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Time")

        fig.colorbar(artist, ax=ax)
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


def plot_consumer_quantity(model: "Model", **kwargs) -> None:
    t = model.time_grid

    plt.plot(t, model.consumer_quantity, **kwargs)
    plt.xlabel("Time")
    plt.ylabel("Consumer quantity")
    plt.ylim(bottom=0.0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_kernel(model: "Model", kernel_format: Literal["heat", "3D"], **kwargs) -> None:
    # Grafico 3D o de calor, para plotear kernels
    return


def plot_1D(
    model: "Model",
    function: Literal["c-growth", "c-decay", "r-growth", "r-decay"],
    **kwargs,
) -> None:
    return


class Plot:
    def __init__(self, parent_model: "Model") -> None:
        self.model = parent_model

    def solution_over_time(
        self,
        plot_type: Literal["heatmap", "contour"] = "heatmap",
        **kwargs,
    ) -> None:
        plot_solution_1D(self.model, plot_type, **kwargs)

    def consumer_quantity(self, **kwargs) -> None:
        plot_consumer_quantity(self.model, **kwargs)

    def kernel(self, form: Literal["heat", "3D"], **kwargs) -> None:
        plot_kernel(self.model, form, **kwargs)

    def unidimensional_function(
        self,
        solution: Literal["c-growth", "c-decay", "r-growth", "r-decay"],
        **kwargs,
    ) -> None:
        plot_1D(self.model, solution, **kwargs)

    def solution_2D_animation(
        self,
        solution: Literal["consumer", "resource"],
        interval: int = 40,
        **kwargs,
    ) -> HTML:
        """
        Anima la evolución temporal de una solución definida sobre un
        espacio bidimensional de rasgos.

        En cada instante de tiempo se grafica una superficie

            z = u(x₁,x₂,t),

        donde los ejes x e y corresponden a los dos rasgos y el eje z
        representa la densidad de consumidores o recursos.

        Parameters
        ----------
        model : Model
            Modelo ya resuelto.

        solution : {"consumer", "resource"}
            Solución a visualizar.

        interval : int, default=40
            Tiempo entre cuadros consecutivos en milisegundos.

        **kwargs
            Argumentos adicionales pasados a
            ``Axes3D.plot_surface``.
        """
        model = self.model
        _require_dimension(model, dimensions=2)

        if solution == "consumer":
            grid = model.consumer_grid
            values = model.consumer_distribution
        else:
            grid = model.resource_grid
            values = model.resource_distribution

        shape = grid.shape

        X = grid.points[:, 0].reshape(shape)
        Y = grid.points[:, 1].reshape(shape)

        zmin = float(np.min(values))
        zmax = float(np.max(values))

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection="3d")

        ax.set_xlabel("Trait 1")
        ax.set_ylabel("Trait 2")
        ax.set_zlabel("Density")
        ax.set_zlim(zmin, zmax)

        surface: Poly3DCollection = ax.plot_surface(
            X, Y, values[0].reshape(shape), vmin=zmin, vmax=zmax, **kwargs
        )

        def update(frame: int):
            nonlocal surface
            surface.remove()

            Z = values[frame].reshape(shape)
            surface = ax.plot_surface(
                X, Y, Z, vmin=zmin, vmax=zmax, **kwargs, color="royalblue"
            )

            ax.set_title(f"t = {model.time_grid[frame]:.3f}")

            if frame % 100 == 0:
                print(f"Frame: {frame}.")

            return (surface,)

        animation = FuncAnimation(
            fig, update, frames=model.n_t, interval=interval, repeat=True
        )
        plt.close(fig)

        return HTML(animation.to_html5_video())
