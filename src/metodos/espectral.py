from typing import TYPE_CHECKING

import numpy as np
import scipy as sp

if TYPE_CHECKING:
    from src.model import Model


# placeholder
def solve_model_by_spectral(model: "Model", *args, **kwargs):
    n = 1
    A = np.zeros(n)
    return A, A
