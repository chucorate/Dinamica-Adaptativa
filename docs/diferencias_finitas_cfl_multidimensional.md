
# Análisis de Estabilidad: Condiciones CFL (Caso Multidimensional)

## Nota

Esta versión está concebida directamente para una dimensión espacial arbitraria $d$, de modo que el caso unidimensional se obtiene tomando $d=1$. La única hipótesis es una malla cartesiana uniforme, con paso $h_m$ en cada dirección. La versión unidimensional se encuentra [aquí](docs/diferencias_finitas_cfl.md).

## Cambios conceptuales respecto de la versión unidimensional

En una dimensión, una malla uniforme se escribe como $$x_j=x_{\min}+jh,$$ donde el índice entero $j$ identifica el nodo de la malla.

En dimensión arbitraria $d$, el espacio pasa a ser

$$\mathbf x=(x_1,\ldots,x_d)\in\mathbb R^d.$$

Cada dirección espacial posee su propia discretización, por lo que los nodos quedan definidos mediante multiíndices

$$\mathbf j=(j_1,\ldots,j_d),$$

donde $j_r$ representa el número de nodo en la dirección $r-$ésima. La posición del nodo correspondiente es

$$\mathbf x_{\mathbf j}=(x_{1,j_1},\ldots,x_{d,j_d}),$$

con

$$x_{r,j_r}=x_{r,\min}+j_rh_r,\qquad r=1,\ldots,d,$$

donde $h_r$ es el paso de malla en la dirección $r$.

En una dimensión, un modo de Fourier tiene la forma $e^{2\pi i\xi x}$ y queda completamente determinado por una única frecuencia escalar $\xi$. Sin embargo, en varias dimensiones una onda puede oscilar de manera distinta en cada dirección espacial. Por ello, se introduce el vector de frecuencias

$$\boldsymbol{\xi}=(\xi_1,\ldots,\xi_d),$$

cuyas componentes indican la frecuencia de oscilación en cada coordenada del espacio. Al aplicar el criterio de Nyquist a cada coordenada (ver documento unidimensional), entonces cada componente satisface

$$\xi_r\in\left[-\frac1{2h_r},\frac1{2h_r}\right],\qquad r=1,\ldots,d.$$

En consecuencia, el ansatz de von Neumann utilizado en una dimensión,

$$n_j^k=\hat n^ke^{2\pi i\xi x_j},$$

se generaliza de manera natural reemplazando el producto $\xi x$ por el producto interno entre el vector de frecuencias y el vector posición,

$$\boldsymbol{\xi}\cdot\mathbf x_{\mathbf j}=\sum_{r=1}^d\xi_rx_{r,j_r}.$$

Por lo tanto, el modo de Fourier en dimensión arbitraria adopta la forma

$$n_{\mathbf j}^k=\hat n^ke^{2\pi i\sum_{r=1}^d\xi_rx_{r,j_r}},$$

que coincide exactamente con la expresión unidimensional cuando $d=1$.

## Símbolo discreto del operador
Definimos el operador difusivo

$$L=-\varepsilon\Delta_h,$$

donde $\Delta_h$ denota el laplaciano discreto sobre una malla cartesiana. En dimensión $d$, este operador puede escribirse como la suma de las segundas diferencias centradas en cada dirección espacial,

$$\Delta_h=\sum_{r=1}^d\delta_{rr},$$

donde

$$(\delta_{rr}n)_{\mathbf j}=\frac{n_{\mathbf j-\mathbf e_r}-2n_{\mathbf j}+n_{\mathbf j+\mathbf e_r}}{h_r^2}.$$

Aquí, $\mathbf e_r$ representa el $r-$ésimo vector canónico de $\mathbb R^d$. En consecuencia, los índices $\mathbf j\pm\mathbf e_r$ corresponden a los vecinos inmediatos del nodo $\mathbf j$ en dicha dirección.

Al aplicar este operador sobre un modo de Fourier,

$$e^{2\pi i\sum_{r=1}^d\xi_rx_{r,j_r}},$$

cada operador $\delta_{rr}$ actúa únicamente sobre la coordenada $r$, produciendo el mismo factor que en el caso unidimensional,

$$\frac{4}{h_r^2}\sin^2(\pi\xi_rh_r).$$

Como cada operador direccional actúa multiplicando el modo de Fourier por un escalar, y el laplaciano discreto es la suma de dichos operadores, el modo de Fourier continúa siendo un vector propio de $L$. Así,

$$(Ln)_{\mathbf j}=\lambda_L(\boldsymbol{\xi})e^{2\pi i\sum_{r=1}^d\xi_rx_{r,j_r}},$$

donde el símbolo discreto viene dado por

$$\boxed{\lambda_L(\boldsymbol{\xi})=4\varepsilon\sum_{r=1}^d\frac{\sin^2(\pi\xi_rh_r)}{h_r^2}.}$$

Además, como

$$\sin^2(\pi\xi_rh_r)\ge0,$$

para todo $r$, se tiene que

$$\lambda_L(\boldsymbol{\xi})\ge0.$$

El valor máximo del símbolo discreto se obtiene cuando cada término de la suma alcanza su máximo posible, es decir,

$$\sin^2(\pi\xi_rh_r)=1,\qquad r=1,\ldots,d,$$

lo que conduce a

$$\boxed{\lambda_L^{\max}=4\varepsilon\sum_{r=1}^d\frac1{h_r^2}.}$$

Este resultado generaliza el caso unidimensional, recuperándose inmediatamente

$$\lambda_L^{\max}=\frac{4\varepsilon}{h^2}$$

cuando $d=1$.

## Factor de amplificación

Definiendo

$$\rho(\boldsymbol{\xi})=\Delta t\lambda_L(\boldsymbol{\xi}),\qquad\beta=\Delta tg,$$

el factor de amplificación obtenido en el caso unidimensional permanece sin modificaciones y viene dado por

$$\boxed{G(\boldsymbol{\xi})=\frac{1-(1-\theta)\rho}{1+\theta\rho-\beta}.}$$

Por lo tanto, toda la demostración posterior es idéntica a la del caso unidimensional, puesto que depende únicamente de $\rho$.

## Condición de estabilidad

El peor modo verifica

$$\rho_{\max}=4\varepsilon\Delta t\sum_{r=1}^d\frac1{h_r^2}.$$

Para $0\le\theta<\frac{1}{2}$, la condición CFL queda

$$\boxed{\Delta t\le\frac{2}{(1-2\theta)4\varepsilon\displaystyle\sum_{r=1}^d\frac1{h_r^2}-|g|}.}$$

Cuando $\theta\ge\frac{1}{2}$, la estabilidad $\ell^2$ es incondicional para $g<0$, exactamente igual que en una dimensión.

## Condición de positividad

La condición $G(\boldsymbol{\xi})\ge0$ equivale a

$$(1-\theta)\rho_{\max}\le1,$$

es decir,

$$\boxed{\Delta t\le\frac{1}{4(1-\theta)\varepsilon\displaystyle\sum_{r=1}^d\frac1{h_r^2}}.}$$

## Caso isotrópico

Si $h_1=\cdots=h_d=h$, entonces

$$\boxed{\Delta t\le\frac{h^2}{4d(1-\theta)\varepsilon}.}$$

En particular,

- $d=1$

$$\Delta t\le\frac{h^2}{4(1-\theta)\varepsilon},$$

- $d=2$

$$\Delta t\le\frac{h^2}{8(1-\theta)\varepsilon},$$

- $d=3$

$$\Delta t\le\frac{h^2}{12(1-\theta)\varepsilon}.$$

## Observación

La totalidad del análisis de estabilidad, incluyendo el estudio del signo de $G$, la separación entre los casos $\theta\ge1/2$ y $0\le\theta<1/2$, así como la discusión sobre estabilidad y positividad, permanece sin modificaciones respecto del caso unidimensional. Por lo tanto, toda la derivación realizada para el caso unidimensional se extiende de forma inmediata al caso multidimensional sustituyendo el símbolo discreto unidimensional por 

$$\lambda_L(\boldsymbol{\xi})=4\varepsilon\sum_{r=1}^d\frac{\sin^2(\pi\xi_rh_r)}{h_r^2},$$

sin que sea necesario modificar el resto del análisis.
