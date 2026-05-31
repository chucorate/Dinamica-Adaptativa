"""
Submódulo para graficar las soluciones otorgadas por `model`.
"""

from typing import TYPE_CHECKING, Literal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animate

if TYPE_CHECKING:
    from src.model import Model


def plot_solution(
    model: Model,
    solution: Literal["consumer-density", "consumer-quantity", "resource"],
    **kwargs,
) -> None:
    # solucion de la forma u(t,y)
    # en eje x va el rasgo, en eje y va el tiempo (de abajo hacia arriba)
    return


def plot_kernel(model: Model, format: Literal["heat", "3D"], **kwargs) -> None:
    # Grafico 3D o de calor, para plotear kernels
    return


def plot_1D(
    model: Model,
    function: Literal["c-growth, c-decay, r-growth, r-decay"],
    **kwargs,
) -> None:
    return
