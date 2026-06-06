"""
Submódulo para graficar las soluciones otorgadas por `model`.
"""

from typing import TYPE_CHECKING, Literal

import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from src.model import Model


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


def _require_1d(model: "Model") -> None:
    if model.consumer_dimension != 1:
        raise ValueError(
            "Esta visualización requiere d_x = 1.\n"
            f"Dimensión de consumidores actual = {model.consumer_dimension}."
        )

    if model.resource_dimension != 1:
        raise ValueError(
            "Esta visualización requiere d_y = 1.\n"
            f"Dimensión de recursos actual = {model.resource_dimension}."
        )


def plot_solution_1D(
    model: "Model",
    plot_type: Literal["heatmap", "contour"],
    **kwargs,
) -> None:
    _require_1d(model)

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
