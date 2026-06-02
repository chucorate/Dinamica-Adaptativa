# Método de Diferencias Finitas para el Modelo Mutación–Selección–Recurso

## Modelo continuo

Consideramos el sistema

$$\partial_t n(t,x) = \varepsilon \Delta_x n(t,x) + \left(r(x)\int K(x,y)R(t,y)\,dy - m_1(x)\right) n(t,x),$$

donde:
- $n(t,x)$ representa la densidad de consumidores con rasgo $x$.
- $\varepsilon$ es la tasa de mutación.
- $K(x,y)$ modela la interacción entre consumidores y recursos.
- $r(x)$ es la eficiencia de aprovechamiento del recurso.
- $m_1(x)$ es la tasa de mortalidad de los consumidores.

El recurso satisface

$$\partial_t R(t,y) = R_{\mathrm{in}}(y) - R(t,y) \left(m_2(y) + \int r(x)K(x,y)n(t,x)\,dx\right),$$

donde:
- $R(t,y)$ representa la densidad de recursos.
- $R_{\mathrm{in}}(y)$ es la tasa de aporte externo.
- $m_2(y)$ es la tasa de decaimiento natural del recurso.

Además del modelo dinámico, se considera una aproximación cuasi-estacionaria para el recurso, obtenida imponiendo $\partial_t R = 0.$ En dicho caso,

$$R(y) = \frac{R_{\mathrm{in}}(y)}{m_2(y)+\int r(x)K(x,y)n(x)\,dx}.$$

## Modelo Discreto

### Discretización espacial

Introduciendo las mallas uniformes

$$\begin{align*}
x_i &= x_{\min}+ih_x,\qquad i=0,\dots,N_x-1, \\
y_j &= y_{\min}+jh_y,\qquad j=0,\dots,N_y-1,
\end{align*}$$

la solución discreta se denota por

$$n_i^k \approx n(t_k,x_i),\qquad R_j^k \approx R(t_k,y_j).$$

### Aproximación de las integrales

Definimos

$$I(x_i)=\int K(x_i,y)R(y)\,dy,\qquad J(y_j)=\int r(x)K(x,y_j)n(x)\,dx.$$

Ambas integrales se aproximan mediante la regla de Simpson para mayor estabilidad numérica:

$$\begin{align*}
I_i^{k} &\approx \sum_{j=0}^{N_y-1} w_j^{(y)} K(x_i,y_j) R_j^{k}, \\
J_j^{k} &\approx\sum_{i=0}^{N_x-1}w_i^{(x)}r(x_i)K(x_i,y_j)n_i^{k},
\end{align*}$$

donde $w^{(x)}$ y $w^{(y)}$ denotan los vectores ponderación de cada término según la regla de Simpson. En forma matricial, lo anterior es equivalente a calcular

$$I^{k}=K\cdot (w^{(y)} \odot R^{k}), \qquad J^{k}=K^T\cdot\left(w^{(x)}\odot r\odot n^{k}\right),$$

donde $K=(K(x_{i},y_{j}))_{i,j}$, $r=(r(x_{i}))_{i}$, $R^{k}=(R_{j}^{k})_{j}$ y $n^{k}=(n_{i}^{k})_{i}$. Aquí, $\cdot$ denota el producto matricial, mientras que $\odot$ denota producto coordenada a coordenada.

### Discretización del operador de mutación

La difusión se aproxima mediante diferencias finitas centradas. Para un punto interior, consideramos 

$$\Delta_h n(x_i):=\frac{n_{i-1}-2n_i+n_{i+1}}{h_x^2}.$$

Esto genera una matriz tridiagonal $\Delta_h$ tal que $\Delta_h n\approx\Delta_x n$. En la implementación se define el operador

$$L:=-\varepsilon \Delta_h,$$

de modo que $L n \approx -\varepsilon\Delta_x n.$ El signo negativo se introduce para que $L$ sea semidefinida positiva.

### Condiciones de borde 

El problema continuo se define para todo $x,y\in \mathbb{R}$, por lo que en estricto rigor no presenta condiciones de borde. Aún así, para poder simular la ecuación en una malla regular, se imponen las siguientes condiciones de borde posibles:

**1. Neumann**<br>
La condición

$$\frac{\partial n}{\partial x}=0$$

se implementa mediante puntos fantasma reflejados:

$$n_{-1}=n_1,\qquad n_{N_x}=n_{N_x-2}.$$

Esto modifica únicamente la primera y última fila de la matriz $L$.

**2. Periódicas**<br>
La condición

$$n(x_{\min})=n(x_{\max})$$

se implementa conectando el primer y último nodo:

$$L_{0,N_x-1}\neq 0, \qquad L_{N_x-1,0}\neq 0.$$

> Nota: Aún cuando se imponen condiciones de borde, es más conveniente simular el problema sobre un dominio suficientemente grande de manera que el borde afecte lo menos posible la solución en el interior.

## Esquema temporal $\theta$

Una vez definido todo lo anterior, se puede plantear el esquema de resolución. La ecuación del consumidor puede escribirse como

$$\partial_t n=\varepsilon\Delta_x n+g(x,R)n,$$

donde

$$g(x,R)=r(x)\int K(x,y)R(y)\,dy-m_1(x).$$

Definimos además la matriz diagonal

$$G^k=\operatorname{diag}(g(x_0, R^k),\dots,g(x_{N_x-1},R^k)).$$

### Actualización del consumidor

La difusión se discretiza mediante un esquema $\theta$ implícito, mientras que la tasa de crecimiento se evalúa utilizando el recurso conocido en el instante anterior. Usando $L \approx -\varepsilon\Delta_x$, obtenemos

$$\frac{n^{k+1}-n^k}{\Delta t}=-\theta L n^{k+1}-(1-\theta)L n^k+G^k n^{k+1}.$$

Reordenando, se llega a

$$\left(I + \theta\Delta t\,L-\Delta t\,G^k\right)n^{k+1} = \left(I - (1- \theta)\Delta t\,L\right)n^k.$$

Definiendo

$$B_\theta=I-(1-\theta)\Delta t\,L,$$

el sistema lineal a resolver en cada paso temporal es

$$\left(I+\theta\Delta t\,L-\Delta t\,G^k\right)n^{k+1}=B_\theta n^k.$$

Notar que $B_{\theta}$ no depende del tiempo, por lo que dicha matriz se puede precalcular. 

### Actualización del recurso dinámico

Cuando el recurso evoluciona dinámicamente, se utiliza Euler explícito. La ecuación continua se puede escribir como 

$$\partial_t R=R_{\mathrm{in}}-R(m_2+J).$$

La discretización temporal produce

$$R^{k+1}=R^k+\Delta t\left[R_{\mathrm{in}}-R^k(m_2+J^k)\right].$$

### Actualización del recurso estacionario

Cuando se utiliza la aproximación cuasi-estacionaria, el recurso no requiere resolver una ecuación diferencial. Una vez calculado $n^{k+1}$, se evalúa directamente

$$R^{k+1}(y)=\frac{R_{\mathrm{in}}(y)}{m_2(y)+\int r(x)K(x,y)n^{k+1}(x)\,dx}.$$

Por lo tanto, el recurso queda sincronizado con la distribución de consumidores recién calculada.

## Algoritmo implementado

El algoritmo entonces sigue los siguientes pasos:
1. Construir las mallas espaciales y temporal.
2. Construir los pesos de Simpson.
3. Construir la matriz discreta de difusión $L$.
4. Evaluar todas las funciones del modelo sobre las mallas.
5. Inicializar $n^0$.
6. Inicializar $R^0$ (depende de si es dinámico o estacionario).
7. Para cada paso temporal:<br>
   - Calcular las integrales.<br>
   - Construir la tasa de crecimiento $g^k$.<br>
   - Resolver el sistema lineal del esquema $\theta$ para obtener $n^{k+1}$.<br>
   - Actualizar el recurso mediante Euler explícito o mediante la fórmula estacionaria.<br>
8. Almacenar las trayectorias completas de consumidores y recursos.


