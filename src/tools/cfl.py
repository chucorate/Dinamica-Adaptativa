from dataclasses import dataclass
import warnings

import numpy as np

from src.tools.grid import TraitGrid


@dataclass(frozen=True)
class CFLReport:
    """
    Resultado del análisis CFL para el esquema θ.

    Attributes
    ----------
    delta_t : float
        Paso temporal utilizado.

    positivity_limit : float
        Máximo paso temporal permitido por la condición
        de positividad.

    stability_limit : float | None
        Máximo paso temporal permitido por la condición
        de estabilidad. Vale ``None`` cuando dicha
        condición no aplica (por ejemplo θ >= 1/2).

    positivity_ok : bool
        Indica si se cumple la condición de positividad.

    stability_ok : bool | None
        Indica si se cumple la condición de estabilidad.
        Vale ``None`` cuando ésta no fue evaluada.
    """

    delta_t: float
    positivity_limit: float
    stability_limit: float | None

    positivity_ok: bool
    stability_ok: bool | None


def check_cfl_conditions(
    grid: TraitGrid,
    mutation_rate: float,
    delta_t: float,
    theta: float,
    max_abs_growth: float | None = None,
) -> CFLReport:
    """
    Evalúa las condiciones CFL multidimensionales
    asociadas al esquema θ.

    Se verifica la condición de positividad

        Δt <= 1 / (4(1-θ)ε Σ_r 1/h_r²)

    y, cuando θ < 1/2 y se proporciona una cota para
    |g|, también la condición de estabilidad

        Δt <= 2 / ((1-2θ)4ε Σ_r 1/h_r² - |g|).

    Parameters
    ----------
    grid : TraitGrid
        Malla espacial utilizada para discretizar
        el operador difusivo.

    mutation_rate : float
        Tasa de mutación ε.

    delta_t : float
        Paso temporal.

    theta : float
        Parámetro del esquema θ.

    max_abs_growth : float | None, optional
        Cota superior para |g|. Si es ``None``,
        la condición de estabilidad no se evalúa.

    Returns
    -------
    CFLReport
        Resultado completo del análisis CFL.

    """

    inv_h2_sum = np.sum(1.0 / np.square(grid.spacing))

    if mutation_rate <= 0:
        positivity_limit = np.inf
    else:
        positivity_limit = (
            1.0 / (4.0 * (1.0 - theta) * mutation_rate * inv_h2_sum)
            if theta < 1.0
            else np.inf
        )

    positivity_ok = delta_t <= positivity_limit

    if theta >= 0.5 or max_abs_growth is None:
        stability_limit = None
        stability_ok = None
    else:
        denominator = (
            1.0 - 2.0 * theta
        ) * 4.0 * mutation_rate * inv_h2_sum - max_abs_growth

        if denominator <= 0:
            stability_limit = np.inf
        else:
            stability_limit = 2.0 / denominator

        stability_ok = delta_t <= stability_limit

    return CFLReport(
        delta_t=delta_t,
        positivity_limit=positivity_limit,
        stability_limit=stability_limit,
        positivity_ok=positivity_ok,
        stability_ok=stability_ok,
    )


def check_cfl_report(report: CFLReport) -> None:
    """
    Valida un reporte CFL.

    Lanza un warning si alguna de las
    condiciones evaluadas no se satisface.
    """
    if not report.positivity_ok:
        warnings.warn(
            "La condición CFL de positividad no se cumple.\n"
            f"Δt utilizado : {report.delta_t}\n"
            f"Δt máximo    : {report.positivity_limit}",
            RuntimeWarning,
            stacklevel=2,
        )

    if report.stability_ok is not None and not report.stability_ok:
        warnings.warn(
            "La condición CFL de estabilidad no se cumple.\n"
            f"Δt utilizado : {report.delta_t}\n"
            f"Δt máximo    : {report.stability_limit}",
            RuntimeWarning,
            stacklevel=2,
        )
