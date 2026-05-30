"""
Submódulo para graficar las soluciones otorgadas por `model`.
"""

from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animate

if TYPE_CHECKING:
    from src.model import Model
