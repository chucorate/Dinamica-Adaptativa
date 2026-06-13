"""
Submódulo para graficar las soluciones otorgadas por `model`.
"""

from typing import TYPE_CHECKING, Literal, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from IPython.display import HTML

if TYPE_CHECKING:
    from src.model import Model


def _require_dimension(model: "Model", dimensions: int) -> None:
    if not isinstance(dimensions, int) or dimensions < 1:
        raise ValueError(
            f"dimensions must be positive int, got {dimensions}"
        )

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


class Plot:
    def __init__(self, parent_model: "Model") -> None:
        self.model = parent_model

    def solution_over_time_1D(
        self,
        plot_type: Literal["heatmap", "contour"] = "heatmap",
        **kwargs,
    ) -> None:
        model = self.model
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

    def consumer_quantity(self, **kwargs) -> None:
        model = self.model
        t = model.time_grid

        plt.plot(t, model.consumer_quantity, **kwargs)
        plt.xlabel("Time")
        plt.ylabel("Consumer quantity")
        plt.ylim(bottom=0.0)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def kernel(self, form: Literal["heat", "3D"], **kwargs) -> None:
        """
        Grafica el kernel de interacción K(x, y) entre consumidores y recursos.
        """
        model = self.model

        if model.consumer_dimension != 1 or model.resource_dimension != 1:
            raise NotImplementedError(
                "Plotear el kernel de momento sólo es soportado para d_x=d_y=1."
            )

        # 1. Obtenemos las mallas de los rasgos usando las funciones generales
        x = model.consumer_grid.points[:, 0]
        y = model.resource_grid.points[:, 0]

        # 2. Creamos la grilla bidimensional
        # Usamos indexing='ij' para que X tenga forma (n_x, n_y) e Y (n_x, n_y)
        X, Y = np.meshgrid(x, y, indexing="ij")

        # 3. Evaluamos el kernel sobre toda la grilla
        Z = model.resource_consumer_kernel(X, Y)

        # 4. Generamos el gráfico según el formato elegido
        if form == "heat":
            # Parámetros por defecto para el mapa de calor
            default_kwargs: dict[str, Any] = dict(
                aspect="auto",
                origin="lower",
                # extent asigna los límites de los ejes: [xmin_y, xmax_y, ymin_x, ymax_x]
                # En imshow, el eje horizontal es el segundo eje de la matriz (y) y el vertical el primero (x)
                extent=[y.min(), y.max(), x.min(), x.max()],
                cmap="viridis",
            )
            default_kwargs.update(kwargs)

            plt.imshow(Z, **default_kwargs)
            plt.xlabel("Resource trait (y)")
            plt.ylabel("Consumer trait (x)")
            plt.title("Interaction Kernel K(x, y)")
            plt.colorbar(label="Interaction strength")

        elif form == "3D":
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            default_kwargs = dict(cmap="viridis")
            default_kwargs.update(kwargs)

            surf = ax.plot_surface(X, Y, Z, **default_kwargs)
            ax.set_xlabel("Consumer trait (x)")
            ax.set_ylabel("Resource trait (y)")
            ax.set_zlabel("Interaction K(x, y)")
            plt.title("Interaction Kernel 3D")

            # Agregamos una barra de color ajustada al tamaño del gráfico
            fig.colorbar(
                surf, ax=ax, shrink=0.5, aspect=5, label="Interaction strength"
            )

        else:
            raise ValueError(f"Unknown format '{form}'")

        plt.show()

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

            # if frame % 100 == 0:
            #    print(f"Frame: {frame}.")

            return (surface,)

        animation = FuncAnimation(
            fig, update, frames=model.n_t, interval=interval, repeat=True
        )
        plt.close(fig)

        return HTML(animation.to_html5_video())
