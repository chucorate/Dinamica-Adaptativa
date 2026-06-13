# Dinámica Adaptativa

Simulador numérico para estudiar la evolución de poblaciones estructuradas por rasgos mediante dinámica adaptativa. Este proyecto implementa métodos de resolución numérica (diferencias finitas y métodos espectrales) para ecuaciones diferenciales parciales no locales con mutación y selección.

## ¿Qué es la dinámica adaptativa?

La dinámica adaptativa es un marco teórico que describe la evolución de rasgos (traits) cuantitativos en poblaciones mediante la selección natural. El modelo acoplado resuelto aquí incluye:

- **Consumidores**: Distribución de densidad `n(x,t)` con rasgo `x`
- **Recursos**: Distribución de densidad `R(y,t)` con rasgo `y`
- **Dinámicas**: Mutación, crecimiento, mortalidad, e interacción consumidor-recurso

El sistema resulta en un conjunto de ecuaciones diferenciales parciales no locales que se resuelve numéricamente.

## Instalación

### Requisitos previos

- Python 3.13+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recomendado)

### Pasos de instalación

Si `uv` no está instalado, checkea [las siguientes instrucciones de instalación](https://docs.astral.sh/uv/getting-started/installation/).

Para instalar las dependencias del proyecto, ejecutar en la raíz del repositorio el siguiente comando:

```bash
uv sync --dev
```

Esto va a crear el entorno virtual `dinamica-adaptativa`, el cuál debe utilizarse para revisar tipado estático de los scripts de Python, y para ejecutar los jupyter notebooks.

### Alternativa: Anaconda/Miniconda

Alternativamente, es posible usar entornos globales como Anaconda/Miniconda. En tal caso, el entorno de ejecución junto a sus dependencias se tendrán que instalar manualmente.

## Uso

### Ejemplo básico

```python
import numpy as np
from src.model import Model

# Definir dominios de rasgos
consumer_domain = np.array([[0.0, 1.0]])  # Un rasgo de consumidor en [0,1]
resource_domain = np.array([[0.0, 1.0]])  # Un rasgo de recurso en [0,1]

# Crear modelo
model = Model(
    consumer_domain=consumer_domain,
    resource_domain=resource_domain,
    mutation_rate=0.01,
)

# Definir funciones del modelo (tasa de crecimiento, mortalidad, etc.)
model.set_consumer_growth_rate(lambda x: np.ones(x.shape[0]))
model.set_consumer_decay(lambda x: 0.1 * np.ones(x.shape[0]))
model.set_resource_supply_rate(lambda y: np.ones(y.shape[0]))
model.set_resource_decay(lambda y: 0.1 * np.ones(y.shape[0]))

# Definir kernel de interacción
def kernel(x, y):
    # Gaussian kernel
    return np.exp(-((x[:, 0:1] - y[:, 0:1].T) ** 2) / 0.1)

model.set_resource_consumer_kernel(kernel)

# Condiciones iniciales
model.set_initial_data(
    initial_consumer_distribution=lambda x: np.exp(-10 * (x[:, 0] - 0.5) ** 2),
    initial_resource_distribution=lambda y: np.ones(y.shape[0]),
)

# Resolver por diferencias finitas
model.solve_by_finite_differences(
    T=10.0,            # Tiempo final
    n_t=100,           # Número de pasos temporales
    n_x=50,            # Resolución espacial (consumidores)
    n_y=50,            # Resolución espacial (recursos)
    border_type="periodic",
    theta=0.5,         # Crank-Nicolson
    use_stationary_resource=False,
)

# Visualizar resultados
model.plot.solution_over_time_1D(plot_type="heatmap")
model.plot.consumer_quantity()
```

## API principal

### Clase `Model`

**Constructor:**
```python
Model(consumer_domain, resource_domain, mutation_rate)
```

**Métodos de configuración:**
- `set_consumer_growth_rate(function)`: Tasa de crecimiento r(x)
- `set_consumer_decay(function)`: Tasa de mortalidad m₁(x)
- `set_resource_supply_rate(function)`: Flujo de entrada Rin(y)
- `set_resource_decay(function)`: Decaimiento m₂(y)
- `set_resource_consumer_kernel(function)`: Kernel de interacción K(x,y)
- `set_initial_data(n0, R0)`: Condiciones iniciales

**Método de resolución:**
```python
solve_by_finite_differences(
    T,                      # Tiempo final
    n_t,                    # Pasos temporales
    n_x,                    # Resolución consumidores
    n_y,                    # Resolución recursos
    border_type,            # "neumann" o "periodic"
    theta,                  # 0 (explícito), 0.5 (CN), 1 (implícito)
    use_stationary_resource # bool
)
```

**Acceso a resultados:**
- `model.consumer_distribution`: Array (n_t, N_x) con densidad de consumidores
- `model.resource_distribution`: Array (n_t, N_y) con densidad de recursos
- `model.consumer_quantity`: Array (n_t,) con cantidad total de consumidores

**Visualización:**
- `model.plot.solution_over_time_1D()`: Heatmap o contour de la evolución
- `model.plot.consumer_quantity()`: Gráfico de cantidad de consumidores
- `model.plot.kernel()`: Visualización del kernel de interacción
- `model.plot.solution_2D_animation()`: Animación 3D para caso 2D

## Documentación técnica

Para detalles sobre análisis de estabilidad CFL, condiciones de borde, y discretización, ver `docs/`.

# Planeación

- [x] Resolver esquema general unidimensional.
- [x] Crear funciones para plotear las funciones del modelo y la solución.
- [ ] Estudiar monomorficidad y dimorficidad. ***[en progreso]***
- [x] Probar distintos métodos de diferencias finitas (explícito, implícito y semi-implícito).
- [x] Estudiar CFL.
- [ ] Probar métodos espectrales. ***[en progreso]***
- [ ] Simular canibalismo.
- [x] Replicar los resultados del paper en el caso gaussiano.
- [x] Resolver esquema general bidimensional.
- [ ] Ver qué pasa al cambiar condiciones en tiempos arbitrarios. (propuesto)
- [ ] Estudiar polimorficidad. (propuesto)

