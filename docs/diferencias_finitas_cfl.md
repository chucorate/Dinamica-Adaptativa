# Análisis de Estabilidad: Condiciones CFL

## Contexto

Este documento estudia las condiciones de estabilidad temporal para el esquema $\theta$ descrito en la documentación del método de diferencias finita (ver [docs/diferencias_finitas_algoritmo.md](docs/diferencias_finitas_algoritmo.md)). A partir del sistema discreto ya construido, se determinan las restricciones sobre el paso temporal $\Delta t$, es decir, las **condiciones CFL**, que garantizan que los errores numéricos no crezcan a lo largo de la integración. 

El análisis se apoya en la transformada de Fourier para derivar el factor de amplificación de cada modo del esquema. Para ello, se adopta la hipótesis de coeficientes congelados, suponiendo que durante un paso temporal la función de crecimiento $g(x,R)$ puede considerarse aproximadamente constante en una vecindad del punto de análisis, lo que permite usar Fourier a pesar de la dependencia espacial de $g$.

## 1. La Transformada de Fourier como Herramienta de Análisis

### 1.1 Convención adoptada y propiedades relevantes

Se utiliza la transformada de Fourier:

$$\hat{f}(\xi) := \mathcal{F}{[f]}(\xi) = \int_{-\infty}^{\infty} f(x)  e^{-2\pi i \xi x} \, dx,$$

donde $\xi \in \mathbb{R}$ representa la frecuencia en ciclos por unidad de longitud. La transformada inversa correspondiente es:

$$f(x) = {\mathcal{F}^{-1}[\hat{f}]}(x) = \int_{-\infty}^{\infty} \hat{f}(\xi)  e^{2\pi i \xi x} \, d\xi.$$

Las dos propiedades relevantes para el análisis son:

**Propiedad de derivación:** Para $f$ suficientemente regular,

$${\mathcal{F}[\partial_x f]}(\xi) = 2\pi i \xi  \hat{f}(\xi),$$

de donde, aplicando la propiedad dos veces:

$${\mathcal{F}[\partial_{xx} f]}(\xi) = (2\pi i \xi)^2 \hat{f}(\xi) = -4\pi^2\xi^2  \hat{f}(\xi).$$

**Teorema de Parseval:**

$$\lVert f \rVert _{L^2(\mathbb{R})}^2 = \lVert \hat{f} \rVert_{L^2(\mathbb{R})}^2,$$

de modo que controlar la norma de $\hat{f}$ en el espacio de frecuencias equivale a controlar la norma de $f$ en el espacio físico.

### 1.2 El símbolo del operador continuo

Para motivar el análisis del esquema discreto, se aplica la transformada a la ecuación del consumidor con coeficientes constantes $g$ y $\varepsilon$:

$$\partial_t n = \varepsilon  \partial_{xx}n + g  n.$$

Tomando la transformada de Fourier respecto de $x$, y usando la propiedad de derivación:

$$\partial_t \hat{n}(\xi,t) = \varepsilon  {\mathcal{F}[\partial_{xx}n]}(\xi,t) + g  \hat{n}(\xi,t) = \varepsilon\bigl(-4\pi^2\xi^2\bigr)\hat{n}(\xi,t) + g  \hat{n}(\xi,t).$$

Esto se reduce a una EDO escalar en $\hat{n}$ para cada frecuencia fija $\xi$:

$$\partial_t \hat{n}(\xi,t) = \underbrace{\bigl(-4\pi^2\varepsilon\xi^2 + g\bigr)}_{\sigma(\xi)}\hat{n}(\xi,t),$$

con solución exacta:

$$\hat{n}(\xi,t) = e^{\sigma(\xi)  t}  \hat{n}(\xi,0).$$

Cada frecuencia crece o decrece según el signo de $\sigma(\xi)$: para $g < 0$, se tiene $\sigma(\xi) < 0$ para toda $\xi$, y la solución decae en todas sus componentes frecuenciales. La transformada de Fourier convierte el problema de estabilidad de una EDP en una colección de problemas escalares independientes, uno por frecuencia.

### 1.3 Ansatz de von Neumann: extensión al caso discreto

Para una malla uniforme $x_j = x_{\min} + j h_x$ con $j = 0,1,\ldots,N_x-1$, la representación discreta análoga a la transformada inversa es:

$$n_j = \hat{n}\cdot e^{2\pi i \xi x_j} = \hat{n}\cdot e^{2\pi i \xi j h_x},$$

donde $\hat{n} \in \mathbb{C}$ es la amplitud del modo de frecuencia $\xi$. Esta expresión corresponde a un modo de la transformada de Fourier discreta, coherente con el kernel $e^{-2\pi i\xi x}$ de la definición de la transformada.

Investigando, fue posible hallar que 

$$\xi \in \left[-\frac{1}{2h_x}, \frac{1}{2h_x}\right].$$

A partir de que, en una malla de paso $h_x$, la frecuencia máxima representable sin aliasing está dada por el criterio de Nyquist: dos puntos por ciclo como mínimo, lo que impone $1/|\xi|\geq  2h_x$.

Frecuencias $|\xi| > 1/(2h_x)$ no son distinguibles de frecuencias menores en la malla: el análisis cubre exhaustivamente todas las oscilaciones representables cuando $\xi$ recorre este rango.

**Definición del factor de amplificación.** Se asume que la solución discreta tiene la forma $n_j^k = \hat{n}^k e^{2\pi i \xi j h_x}$ y se sustituye en el esquema lineal. Por linealidad, la exponencial $e^{2\pi i\xi j h_x}$ se cancela en ambos lados, resultando en lo siguiente:

$$\hat{n}^{k+1} = G(\xi)\hat{n}^k.$$

Iterando desde el instante inicial: $\hat{n}^k = G(\xi)^k\hat{n}^0$, de donde $|\hat{n}^k| = |G(\xi)|^k|\hat{n}^0|$.

**Criterio de estabilidad** Para que los errores no crezcan, la amplitud de cada modo no debe aumentar con los pasos temporales:

$$\boxed{|G(\xi)| \leq 1 \qquad \forall  \xi \in \left[-\frac{1}{2h_x},\frac{1}{2h_x}\right].}$$

## 2. Estabilidad del Esquema $\theta$ para el Consumidor

El esquema $\theta$ para la ecuación del consumidor es:

$$\left(I + \theta\Delta t  L - \Delta t  G^k\right)n^{k+1} = B_\theta  n^k,\qquad B_\theta = I-(1-\theta)\Delta t  L,$$

donde $L = -\varepsilon\Delta_h$ es el operador de difusión discreto (semidefinido positivo) y $G^k = \mathrm{diag}(g(x_0,R^k),\ldots,g(x_{N_x-1},R^k))$. La fracción $\theta$ de la difusión se trata implícitamente (usa $n^{k+1}$) y la fracción $(1-\theta)$ explícitamente (usa $n^k$). La reacción $G^k n^{k+1}$ siempre usa $n^{k+1}$ con los coeficientes del paso anterior.

### 2.1 Símbolo discreto del operador $L$

El primer paso es calcular la acción de $L$ sobre el modo $e^{2\pi i\xi j h_x}$. La acción de $L = -\varepsilon\Delta_h$ en un nodo interior $j$ es:

$$(Ln)_j = -\varepsilon  \frac{n_{j-1} - 2n_j + n_{j+1}}{h_x^2}.$$

Sustituyendo $n_j = e^{2\pi i\xi j h_x}$:

$$(Ln)_j = -\varepsilon  \frac{e^{2\pi i\xi(j-1)h_x} - 2  e^{2\pi i\xi j h_x} + e^{2\pi i\xi(j+1)h_x}}{h_x^2}.$$

Se factoriza $e^{2\pi i\xi j h_x}$:

$$(Ln)_j = -\varepsilon  e^{2\pi i\xi j h_x}\cdot\frac{e^{-2\pi i\xi h_x} - 2 + e^{2\pi i\xi h_x}}{h_x^2}.$$

La expresión en el numerador se simplifica usando $e^{i\theta} + e^{-i\theta} = 2\cos\theta$ con $\theta = 2\pi\xi h_x$:

$$e^{-2\pi i\xi h_x} - 2 + e^{2\pi i\xi h_x} = 2\cos(2\pi\xi h_x) - 2.$$

Aplicando la identidad trigonométrica $\cos\theta - 1 = -2\sin^2(\theta/2)$ con $\theta = 2\pi\xi h_x$:

$$2\cos(2\pi\xi h_x) - 2 = -4\sin^2(\pi\xi h_x).$$

Sustituyendo todo lo anterior:

$$(Ln)_j = -\varepsilon  e^{2\pi i\xi j h_x}\cdot\frac{-4\sin^2(\pi\xi h_x)}{h_x^2} = \underbrace{\frac{4\varepsilon}{h_x^2}\sin^2(\pi\xi h_x)}_{\lambda_L(\xi)}\cdot e^{2\pi i\xi j h_x}.$$

El modo $e^{2\pi i\xi j h_x}$ es vector propio de $L$ con valor propio:

$$\boxed{\lambda_L(\xi) = \frac{4\varepsilon}{h_x^2}\sin^2(\pi\xi h_x) \geq 0.}$$

El símbolo es no negativo en todo $\xi$ (consistente con $L$ semidefinida positiva) y alcanza su máximo en $|\xi| = 1/(2h_x)$, donde $\sin^2(\pi\cdot\frac{1}{2h_x}\cdot h_x)= \sin^2(\pi/2) = 1$:

$$\lambda_L^{\max} = \frac{4\varepsilon}{h_x^2}.$$

### 2.2 Derivación del factor de amplificación

Con coeficientes congelados $g^k = g$, el operador $G^k = gI$ actúa como multiplicación escalar. Se sustituye $n_j^k = \hat{n}^k e^{2\pi i\xi j h_x}$ en el esquema $\theta$:

- **Lado izquierdo.** Usando que $L  e^{2\pi i\xi j h_x} = \lambda_L(\xi)  e^{2\pi i\xi j h_x}$ y que $G^k  e^{2\pi i\xi j h_x} = g  e^{2\pi i\xi j h_x}$:

$$\bigl(I + \theta\Delta t  L - \Delta t  G^k\bigr)n^{k+1} = \bigl(1 + \theta\Delta t  \lambda_L(\xi) - \Delta t  g\bigr)\hat{n}^{k+1}  e^{2\pi i\xi j h_x}.$$

- **Lado derecho:** 

$$B_\theta  n^k = \bigl(I - (1-\theta)\Delta t  L\bigr)n^k= \bigl(1-(1-\theta)\Delta t  \lambda_L(\xi)\bigr)\hat{n}^k  e^{2\pi i\xi j h_x}.$$

Igualando ambos lados y cancelando el factor común $e^{2\pi i\xi j h_x} \neq 0$:

$$\bigl(1 + \theta\Delta t  \lambda_L(\xi) - \Delta t  g\bigr)\hat{n}^{k+1} = \bigl(1-(1-\theta)\Delta t  \lambda_L(\xi)\bigr)\hat{n}^k.$$

Despejando $G(\xi) = \hat{n}^{k+1}/\hat{n}^k$ se obtiene el **factor de amplificación del esquema $\theta$**:

$$\boxed{G(\xi) = \frac{1-(1-\theta)\Delta t  \lambda_L(\xi)}{1+\theta\Delta t  \lambda_L(\xi) - \Delta t  g} = \frac{1-(1-\theta)  \dfrac{4\varepsilon\Delta t}{h_x^2}\sin^2(\pi\xi h_x)}{1+\theta  \dfrac{4\varepsilon\Delta t}{h_x^2}\sin^2(\pi\xi h_x) - \Delta t  g}.}$$

Para condensar la notación se definen los parámetros adimensionales:

$$\rho(\xi) := \Delta t  \lambda_L(\xi) = \frac{4\varepsilon\Delta t}{h_x^2}\sin^2(\pi\xi h_x) \geq 0, \qquad \beta := \Delta t  g,$$

de manera que:

$$G(\xi) = \frac{1-(1-\theta)\rho}{1+\theta\rho - \beta}.$$

Note que $\rho$ crece cuadráticament al hacer la malla más fina o al aumentar $\varepsilon$. El parámetro $\beta$ es negativo y grande en módulo cuando la mortalidad $m_1$ es elevada.

### 2.3 Condición de estabilidad para $g < 0$ (régimen de extinción)

En el modelo de Perthame, el régimen dominante es $g < 0$ (la mortalidad supera el beneficio del recurso), por lo que $\beta < 0$, o equivalentemente $|\beta| = \Delta t|g|$. El denominador satisface:

$$1 + \theta\rho - \beta = 1 + \theta\rho + |\beta| \geq 1 > 0 \qquad \forall  \rho \geq 0, \theta \geq 0.$$

El denominador es siempre positivo: el sistema lineal $\bigl(I+\theta\Delta t L - \Delta t G^k\bigr)n^{k+1} = B_\theta n^k$ tiene solución única para todo $\Delta t > 0$.

La condición $|G(\xi)| \leq 1$ se analiza como dos desigualdades separadas.

#### Cota superior: $G(\xi) \leq 1$

Dado que el denominador es positivo, la condición es equivalente a:

$$1-(1-\theta)\rho \leq 1+\theta\rho+|\beta|.$$

Simplificando el lado izquierdo y el derecho:

$$-(1-\theta)\rho - \theta\rho \leq |\beta| \implies -\rho \leq |\beta|.$$

Esta desigualdad es siempre verdadera puesto que $\rho \geq 0$ y $|\beta| \geq 0$. La cota superior se satisface **incondicionalmente** para cualquier $\theta$, $\Delta t$ y $g < 0$.

#### Cota inferior: $G(\xi) \geq -1$

Dado que el denominador es positivo, la condición equivale a:

$$1-(1-\theta)\rho \geq -(1+\theta\rho+|\beta|).$$

Pasando todos los términos al lado izquierdo:

$$1-(1-\theta)\rho + 1 + \theta\rho + |\beta| \geq 0.$$

Agrupando los términos en $\rho$: 

$$-(1-\theta)\rho + \theta\rho = \bigl(-(1-\theta) + \theta\bigr)\rho = (2\theta-1)\rho.$$

La desigualdad queda:

$$2 + (2\theta-1)\rho + |\beta| \geq 0. \qquad (\star)$$

El análisis de $(\star)$ se separa según el signo del coeficiente de $\rho$.

**Caso $\theta \geq \frac{1}{2}$:** $(2\theta-1) \geq 0$, de modo que cada sumando en $(\star)$ es no
negativo:

$$2 + (2\theta-1)\rho + |\beta| \geq 2 > 0. \qquad$$

Para $g < 0$ y $\theta \geq 1/2$, el esquema es **incondicionalmente estable**. No se impone ninguna
condición sobre $\Delta t$. Este resultado abarca los métodos de Crank-Nicolson ($\theta = 1/2$) y
Euler implícito ($\theta = 1$).

**Caso $0 \leq \theta < \frac{1}{2}$:** $(2\theta-1) < 0$, y $(\star)$ exige:

$$(1-2\theta)\rho \leq 2 + |\beta|.$$

Se sustituye $\rho = \Delta t  \lambda_L(\xi)$ y se acota el lado izquierdo por su máximo sobre todo $\xi$. El máximo de $\rho(\xi)$ sobre el rango ocurre en $|\xi| = 1/(2h_x)$, donde $\sin^2(\pi \cdot \frac{1}{2h_x} \cdot h_x) = \sin^2(\pi/2) = 1$, dando $\rho_{\max} = 4\varepsilon\Delta t/h_x^2$. La condición necesaria sobre el peor modo es:

$$(1-2\theta)\frac{4\varepsilon\Delta t}{h_x^2} \leq 2 + \Delta t|g|.$$

Reordenando los términos en $\Delta t$ al mismo lado:

$$\Delta t\left[(1-2\theta)\frac{4\varepsilon}{h_x^2} - |g|\right] \leq 2.$$

El análisis del signo del coeficiente de $\Delta t$ da dos subcasos:

- **Si $(1-2\theta)\dfrac{4\varepsilon}{h_x^2} \leq |g|$**: el coeficiente es no positivo y la desigualdad se cumple para todo $\Delta t > 0$. **Sin condición CFL.** 

- **Si $(1-2\theta)\dfrac{4\varepsilon}{h_x^2} > |g|$**: el coeficiente es positivo y se puede dividir, obteniendo la condición CFL:

$$\boxed{\Delta t \leq \frac{2}{  (1-2\theta)\dfrac{4\varepsilon}{h_x^2} - |g|  }.} \tag{CFL-$\theta$}$$

**Interpretación de los casos límite.**

- Para $\theta = 0$ (difusión explícita) y $|g| = 0$ (solo difusión): 

$$\Delta t \leq \frac{2}{4\varepsilon/h_x^2} = \frac{h_x^2}{2\varepsilon}.$$

- Para $\theta = 0$ y $|g|$ creciente: el denominador de $(\text{CFL}-\theta)$ disminuye, permitiendo pasos mayores. Cuando $|g| \to 4\varepsilon/h_x^2$, la condición desaparece: la rigidez de la reacción domina la difusión.

- Para $\theta \to 1/2$: el denominador $\to -|g| \leq 0$, y la condición deja de imponer restricción.

### 2.4 Condición de positividad del esquema $\theta$

Más allá de la estabilidad $\ell^2$, para ecuaciones de densidades de población se requiere $n_j^{k+1} \geq 0$ cuando $n_j^k \geq 0$ (positividad del esquema). Dado que el denominador de $G(\xi)$ es positivo para $g < 0$, la condición $G(\xi) \geq 0$ equivale a exigir que el numerador sea no negativo:

$$1 - (1-\theta)\rho \geq 0 \iff (1-\theta)\frac{4\varepsilon\Delta t}{h_x^2}\sin^2(\pi\xi h_x) \leq 1.$$

El caso más restrictivo es la frecuencia de borde $|\xi| = 1/(2h_x)$:

$$\boxed{(1-\theta)\frac{4\varepsilon\Delta t}{h_x^2} \leq 1 \iff \Delta t \leq \frac{h_x^2}{4(1-\theta)\varepsilon}.} \tag{POS-$\theta$}$$

Para $\theta = 1$: la condición desaparece. Para $\theta = 1/2$: 

$$\Delta t \leq h_x^2/(2\varepsilon).$$

Para $\theta = 0$: 

$$\Delta t \leq h_x^2/(4\varepsilon).$$

Comparando $(\text{CFL}-\theta)$ y $(\text{POS}-\theta)$ para $\theta = 0$ con $|g| = 0$: la estabilidad $\ell^2$ exige $\Delta t \leq h_x^2/(2\varepsilon)$, mientras que la positividad exige $\Delta t \leq h_x^2/(4\varepsilon)$. La condición de positividad es el doble de restrictiva, lo que refleja el hecho general de que la positividad del esquema es una propiedad más fuerte que la estabilidad en norma $\ell^2$.

Cuando la positividad se viola pero se mantiene la estabilidad $\ell^2$, es decir, cuando 

$$h_x^2 / (4(1-\theta)\varepsilon) < \Delta t \leq h_x^2 / (2(1-\theta)\varepsilon),$$ 

el factor de amplificación satisface $-1 \leq G(\xi) < 0$: el modo de frecuencia $\xi$ cambia de signo en cada paso temporal y converge a cero, pero no monótonamente. Esto genera oscilaciones espaciales de periodo $2h_x$ que se manifiestan como ruido en la solución.

### 2.5 Caso $g > 0$ (régimen de crecimiento neto)

Cuando $g > 0$, $\beta > 0$ y el denominador $1+\theta\rho - \beta$ puede anularse o cambiar de signo. El caso más crítico es el modo $\xi = 0$ (el modo que controla la cantidad total $\rho(t) = \sum_j n_j^k h_x$), donde $\lambda_L(0) = 0$ y $\rho(0) = 0$:

$$G(0) = \frac{1}{1 - \beta} = \frac{1}{1 - \Delta t  g}.$$

Para $\Delta t  g < 1$: $G(0) > 1$ (crecimiento físico del modo promedio). 

Para $\Delta t  g = 1$: el denominador se anula (singularidad). Para $\Delta t  g > 1$: $G(0) < 0$ (el modo promedio cambia de signo). 

La condición de positividad para el modo $\xi = 0$ exige:

$$\Delta t < \frac{1}{\displaystyle\max_{j,k}(g_j^k)^+},$$

donde $(g_j^k)^+ = \max(g_j^k, 0)$. En el modelo de Perthame, $g > 0$ solo ocurre en la vecindad del ESD donde $g \approx 0$, por lo que esta condición es en la práctica no restrictiva.

## 3. Estabilidad del Recurso Dinámico

El esquema para el recurso dinámico es Euler explícito. La ecuación discreta en cada punto $y_j$, con $\Lambda_j^k = m_2(y_j) + J_j^k$ y $J_j^k$ calculado mediante la regla de Simpson, es:

$$R_j^{k+1} = R_j^k + \Delta t\bigl[R_{in,j} - R_j^k  \Lambda_j^k\bigr] = \underbrace{(1 - \Delta t  \Lambda_j^k)}_{G_R}  R_j^k + \Delta t  R_{in,j}.$$

**Por qué no aplica el ansatz de Fourier.** La ecuación del recurso no contiene ningún operador diferencial en la variable $y$: no hay difusión espacial en la dirección del rasgo del recurso. Los distintos puntos $y_j$ están acoplados únicamente a través de la integral 

$$J_{j}^{k} = \sum_i w_i^{(x)} r(x_i)K(x_i,y_j)n_{i}^{k},$$

que es un parámetro para la recurrencia de $R_j^k$ dado el estado del consumidor en el paso $k$. El análisis se reduce por tanto al estudio de la estabilidad de una **recurrencia escalar lineal** en cada $j$, con coeficiente $G_R = 1 - \Delta t\Lambda_j^k$.

### 3.1 Estabilidad de la recurrencia escalar

Se define la desviación respecto al equilibrio local ${\bar{R}_j^{k}} = {R_{in,j}}/ \Lambda_j^k$:

$$e_j^k := R_j^k - \bar{R}_j^k.$$

La evolución del error satisface, usando $\Delta t  R_{in,j} = \Delta t  \Lambda_j^k  \bar{R}_j^k$:

$$e_j^{k+1} = R_j^{k+1} - \bar{R}_j^k = G_R  R_j^k + \Delta t  R_{in,j} - \bar{R}_j^k = G_R(R_j^k - \bar{R}_j^k) = G_R  e_j^k.$$

El error al paso $k$ es $e_j^k = G_R^k  e_j^0$, y su módulo evoluciona como $|e_j^k| = |G_R|^k  |e_j^0|$.

### 3.2 Condición de estabilidad $\ell^2$

Para que $|e_j^k|$ permanezca acotado: $|G_R| \leq 1$, es decir:

$$|1 - \Delta t  \Lambda_j^k| \leq 1 \iff -1 \leq 1 - \Delta t  \Lambda_j^k \leq 1.$$

La cota superior ($1 - \Delta t\Lambda_j^k \leq 1$) reduce a $\Lambda_j^k \geq 0$, siempre verdadero. La cota inferior ($1 - \Delta t\Lambda_j^k \geq -1$) da $\Delta t\Lambda_j^k \leq 2$:

$$\boxed{\Delta t \leq \frac{2}{\displaystyle\max_{j,k}  \Lambda_j^k}.} \qquad (\text{CFL}-R)$$

### 3.3 Condición de positividad

Para $R_j^{k+1} \geq 0$ cuando $R_j^k \geq 0$ y $R_{in,j} \geq 0$, se necesita $G_R \geq 0$:

$$1 - \Delta t  \Lambda_j^k \geq 0 \iff \boxed{\Delta t \leq \frac{1}{\displaystyle\max_{j,k}  \Lambda_j^k}.} \qquad (\text{POS}-R)$$

Cuando $(\text{POS}-R)$ se viola pero $(\text{CFL}-R)$ se satisface, el factor $G_R$ toma valores en $[-1,0)$: el recurso $R_j^k$ oscila entre valores positivos y negativos en pasos alternos. Como $R$ alimenta el cálculo de $g^k = r(x)\int K(x,y)R^k(y)  dy - m_1(x)$, estos valores negativos dan lugar a un $g^k$ artificialmente reducido (o incluso muy negativo), desestabilizando la ecuación del consumidor a través del acoplamiento.

### 3.4 Estimación de $\Lambda_j^k$

Para cuantificar la condición CFL, se estima $\Lambda_j^k$ al instante inicial con dato uniforme $n_i^0 = n_0$ y parámetros constantes $r(x_i) = r$:

$$\Lambda_j^0 = m_2 + r  n_0\sum_{i=0}^{N_x-1}w_i^{(x)}  K(x_i,y_j) \approx m_2 + r  n_0\int_{x_{\min}}^{x_{\max}} K(x,y_j) \, dx.$$

Para el kernel gaussiano $K(x,y) = \frac{1}{\sigma_K\sqrt{2\pi}}e^{-|x-y|^2/(2\sigma_K^2)}$, la integral se evalúa analíticamente:

$$\int_{x_{\min}}^{x_{\max}}K(x,y_j) \, dx = \frac{1}{2}\left[\mathrm{erf}\!\left(\frac{x_{\max}-y_j}{\sigma_K\sqrt{2}}\right) - \mathrm{erf}\!\left(\frac{x_{\min}-y_j}{\sigma_K\sqrt{2}}\right)\right].$$

Donde $\mathrm{erf}(z)=\frac{2}{\pi}\int_0^ze^{-t^2}dt$. Para dominios amplios respecto de $\sigma_K$, la integral es aproximadamente 1 (toda la gaussiana
está en el dominio) y $\Lambda_j^0 \approx m_2 + r  n_0$.

### 3.5 Caso cuasi-estacionario

Cuando se utiliza la fórmula algebraica $R_j^{k+1} = R_{in,j}/(m_2(y_j) + J_j^{k+1})$, no existe recurrencia temporal para $R$: el recurso queda determinado algebraicamente a partir de $n^{k+1}$ ya calculado. El denominador $m_2(y_j) + J_j^{k+1} \geq m_2(y_j) > 0$ es siempre positivo, garantizando $R_j^{k+1} > 0$. **No se impone ninguna condición CFL sobre el recurso.**

## 4. Resumen: Condiciones CFL por Configuración

La tabla siguiente consolida todas las condiciones identificadas. 

| Origen | Condición | Se activa cuando |
|---|---|---|
| Difusión, $\ell^2$, $\theta \geq 1/2$ | Ninguna | — |
| Difusión, $\ell^2$, $\theta \in [0,1/2)$ | $\Delta t \leq \dfrac{2}{(1-2\theta)\frac{4\varepsilon}{h_x^2} - \|g\|}$ | $(1-2\theta)\frac{4\varepsilon}{h_x^2} > \|g\|$
| Difusión, positividad, $\theta < 1$ | $\Delta t \leq \dfrac{h_x^2}{4(1-\theta)\varepsilon}$ | Siempre |
| Difusión, positividad, $\theta = 1$ | Ninguna | — |
| Recurso dinámico, $\ell^2$ | $\Delta t \leq 2/\Lambda_{\max}$ | Siempre |
| Recurso dinámico, positividad | $\Delta t \leq 1/\Lambda_{\max}$ | Siempre |
| Recurso cuasi-estacionario | Ninguna | — |

**Ejemplo.** El uso de $\theta = 1$ (Euler implícito en difusión) junto con recurso cuasi-estacionario elimina todas las condiciones CFL. El único costo computacional adicional es resolver un sistema tridiagonal con diagonal variable en cada paso, realizable en $O(N_x)$ operaciones mediante el algoritmo de Thomas.

## Apéndice: El Modo $\xi = 0$ y las Oscilaciones en $\rho(t)$

La cantidad total de consumidores $\rho(t_k) = \sum_j n_j^k h_x$ corresponde exactamente al modo $\xi = 0$ de la transformada discreta: para $\xi = 0$, $e^{2\pi i \cdot 0 \cdot j h_x} = 1$ para todo $j$, de modo que la amplitud del modo $\xi = 0$ es $\hat{n}^k = \rho(t_k)/(N_x h_x^{-1})$, proporcional a la masa total.

Para $\xi = 0$: $\lambda_L(0) = \frac{4\varepsilon}{h_x^2}\sin^2(0) = 0$, y por tanto $\rho(0) = 0$. El factor de amplificación del esquema $\theta$ en el modo $\xi = 0$ es:

$$G(0) = \frac{1 - (1-\theta)\cdot 0}{1 + \theta\cdot 0 - \beta} = \frac{1}{1-\beta} = \frac{1}{1 - \Delta t  g}.$$

Para $g < 0$ y cualquier valor de $\theta$: $G(0) = 1/(1+\Delta t|g|) \in (0,1)$. El modo $\xi = 0$ es **incondicionalmente estable en el esquema $\theta$**, independientemente de $\Delta t$.

Esto implica que las oscilaciones en $\rho(t)$ no pueden provenir del modo $\xi = 0$ del esquema $\theta$. Si se observan oscilaciones en $\rho(t)$, las causas posibles son:

1. **Recurso dinámico con $(\text{POS}-R)$ violada:** $\Delta t > 1/\Lambda_{\max}$ hace que $G_R < 0$ y $R_j^k$ tome valores negativos, generando valores de $g^k$ incorrectos que desestabilizan la ecuación del consumidor por acoplamiento.

2. **Dato inicial no resuelto:** si $\sigma_{init} < 5  h_x$, el laplaciano discreto amplifica artificialmente los modos de alta frecuencia del dato inicial, generando oscilaciones transitorias.

3. **Efectos no lineales del acoplamiento:** cuando $\Delta t$ es grande, las variaciones de $R^k$ entre pasos introducen errores en $g^k$ no capturados por el análisis de coeficientes congelados.
