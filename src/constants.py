"""
Constantes globales para el proyecto.

Define tipos de datos y valores constantes utilizados en toda la simulación.
"""

import numpy as np

# Tipo de dato de precisión utilizado en todos los cálculos numéricos
dtype = np.float64

# Valor mínimo permitido para denominadores en cálculos numéricos
# para evitar división por cero o números extremadamente pequeños que causen inestabilidad.
MIN_DENOMINATOR = 1e-12

# Valor mínimo permitido para las densidades de consumidor y recursos
# para evitar comportamientos inesperados por floating points (por ejemplo, que la población reviva luego de cierto tiempo)
MIN_DENSITY = 1e-18
