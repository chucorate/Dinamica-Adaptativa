# Método de Diferencias Finitas para el Modelo Mutación–Selección–Recurso

## Modelo continuo

Consideramos el sistema

$$\partial_t n(t,x) = \varepsilon \Delta_x n(t,x) + \left(r(x)\int K(x,y)R(t,y)\,dy - m_1(x)\right) n(t,x),$$

donde:
- $x\in \mathbb{R}^{d_x},y\in \mathbb{R}^{d_y}$ y $d_x,d_y\in\mathbb{N}$.
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

### Discretización espacio-temporal

El dominio del consumidor y del recurso respectivamente se denotarán como

$$\Omega_x = [a_1,b_1]\times\cdots\times[a_{d_x},b_{d_x}],\quad \Omega_y = [c_1,d_1]\times\cdots\times[c_{d_y},d_{d_y}].$$

Para cada componente $\ell$, se construye una malla uniforme $x^{(\ell)}$ que contiene los puntos

$$x^{(\ell)}_{i_\ell} = a_\ell+\frac{i_\ell}{N_\ell - 1} (b_\ell - a_\ell), \qquad i_\ell=0,\dots,N_\ell-1,$$

que discretiza el intervalo $[a_\ell, b_\ell]$ en $N_\ell$ puntos. Análogamente para el recurso,

$$y^{(\ell)}_{j_\ell} = c_\ell+\frac{j_\ell}{M_\ell - 1} (d_\ell - c_\ell), \qquad j_\ell=0,\dots,M_\ell-1$$

discretiza el intervalo $[c_\ell, d_\ell]$ en $M_\ell$ puntos. Con esto, las grillas de los espacios de rasgos de consumidor y recursos son respectivamente 

$$\mathcal X = x^{(1)}\times\cdots\times x^{(d_x)},\quad \mathcal{Y} = y^{(1)}\times\cdots\times y^{(d_y)},$$

las cuales tienen un total de puntos $N,M$ dados por

$$N=\prod_{\ell=1}^{d_x} N_\ell,\qquad M=\prod_{\ell=1}^{d_y} M_\ell.$$

Por otro lado, para discretizar el tiempo consideramos un tiempo máximo $T$ y un $N_T\in \mathbb{N}$. Con esto, el tiempo se discretiza mediante los puntos

$$t_k = \frac{k}{N_T-1}T,\qquad k=0,\dots,N_T-1.$$

Así, las soluciones discretas vienen dadas por

$$n_i^k \approx n(t_k,x_i), \qquad R_j^k \approx R(t_k,y_j),$$

donde cada punto de la grilla del consumidor y del recurso se identifican mediante los multiíndices

$$i=(i_1,\dots,i_{d_x}),\qquad j=(j_1,\dots,j_{d_y}).$$

En la implementación, ambos multiíndices se linealizan para almacenar las soluciones como matrices de dimensión $(N, d_x)$ y $(M, d_y)$ respectivamente, donde cada fila corresponde a un punto de los espacios de rasgos. En consecuencia, las soluciones discretas se almacenan como matrices de dimensión $(N_T,N)$ y $(N_T,M)$, cuyas filas representas los distintos instantes de tiempo.

### Aproximación de las integrales

Definimos

$$I(x_i)=\int K(x_i,y)R(y)\,dy,\qquad J(y_j)=\int r(x)K(x,y_j)n(x)\,dx.$$

Ambas integrales se aproximan mediante la regla de Simpson para mayor estabilidad numérica:

$$\begin{align*}
I_i^{k} &\approx \sum_{j=0}^{N_y-1} w_j^{(y)} K(x_i,y_j) R_j^{k}, \\
J_j^{k} &\approx\sum_{i=0}^{N_x-1}w_i^{(x)}r(x_i)K(x_i,y_j)n_i^{k},
\end{align*}$$

donde $w^{(x)}$ y $w^{(y)}$ denotan los vectores ponderación de cada término según la regla de Simpson. En el caso multidimensional, dichos pesos se obtienen mediante el producto tensorial de los pesos unidimensionales de cada coordenada.

En forma matricial, lo anterior es equivalente a calcular

$$I^{k}=K\cdot (w^{(y)} \odot R^{k}), \qquad J^{k}=K^T\cdot\left(w^{(x)}\odot r\odot n^{k}\right),$$

donde $K=(K(x_{i},y_{j}))_{i,j}$, $r=(r(x_{i}))_{i}$, $R^{k}=(R_{j}^{k})_{j}$ y $n^{k}=(n_{i}^{k})_{i}$. Aquí, $\cdot$ denota el producto matricial, mientras que $\odot$ denota producto coordenada a coordenada.

### Discretización del operador de mutación

La difusión se aproxima mediante diferencias finitas centradas. En el caso unidimensional, para un punto interior consideramos 

$$\Delta_h n(x_i):=\frac{n_{i-1}-2n_i+n_{i+1}}{h_x^2}.$$

Esto genera una matriz tridiagonal $\Delta_h$ tal que $\Delta_h n\approx\Delta_x n$. Para el caso multidimensional, el laplaciano discreto se construye mediante productos de Kronecker:

$$\Delta_h=\sum_{\ell=1}^{d_x}I_{N_1}\otimes\cdots\otimes I_{N_{\ell-1}}\otimes\Delta_h^{(\ell)}\otimes I_{N_{\ell+1}}\otimes\cdots\otimes I_{N_{d_x}},$$

donde $I_{N_\ell}$ denota la matriz identidad de dimensión $N_\ell$, y $\Delta_h^{(\ell)}$ es el laplaciano unidimensional asociado a la componente $\ell$.

En la implementación se define el operador

$$L:=-\varepsilon \Delta_h,$$

de modo que $L n \approx -\varepsilon\Delta_x n.$ El signo negativo se introduce para que $L$ sea semidefinida positiva.

### Condiciones de borde 

El problema continuo se define para todo $x\in\mathbb{R}^{d_x}$, por lo que en estricto rigor no presenta condiciones de borde. Aún así, para poder simular la ecuación en una malla regular, se imponen las siguientes condiciones de borde posibles:

**1. Neumann**<br>
La condición

$$\frac{\partial n}{\partial x}=0$$

se implementa mediante puntos fantasma reflejados coordenada a coordenada:

$$n_{-1}=n_1,\qquad n_{N_\ell}=n_{N_\ell-2}.$$
Esto modifica únicamente la primera y última fila del laplaciano discreto unidimensional $\Delta_h^{(\ell)}$ asociado a la coordenada $\ell$. El operador multidimensional se obtiene posteriormente mediante productos de Kronecker.

**2. Periódicas**<br>
La condición

$$n(a_{\ell})=n(b_{\ell})$$

se implementa identificando el primer y el último nodo de cada coordenada. En consecuencia, el laplaciano unidimensional $\Delta_h^{(\ell)}$ incorpora las conexiones entre ambos extremos del intervalo mediante entradas no nulas en las esquinas de su matriz:

$$(\Delta_h^{(\ell)})_{0,N_\ell-1}\neq 0, \qquad (\Delta_h^{(\ell)})_{N_\ell-1,0}\neq 0.$$

El operador multidimensional $\Delta_h$ hereda estas conexiones a través de su construcción mediante productos de Kronecker.

> Nota: Aún cuando se imponen condiciones de borde, es más conveniente simular el problema sobre un dominio suficientemente grande de manera que el borde afecte lo menos posible la solución en el interior.

## Esquema temporal $\theta$

Una vez definido todo lo anterior, se puede plantear el esquema de resolución. La ecuación del consumidor puede escribirse como

$$\partial_t n=\varepsilon\Delta_x n+g(x,R)n,$$

donde

$$g(x,R)=r(x)\int_{\Omega_y} K(x,y)R(y)\,dy-m_1(x).$$

Definimos además la matriz diagonal

$$G^k=\mathrm{diag}(g(x_0, R^k),\dots,g(x_{N-1},R^k)).$$

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


