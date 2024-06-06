# MySpline


The mathematical procedure we implemented in GaPSE is based on:
- Parviz Moin, _"Fundamentals of Engineering Numerical Analysis"_ (2010), 
  Cambridge University Press: Second edition, Chapter 1.2, "Cubic Spline Interpolation"
- 

## Derivation of the equation system for the cubic spline

Let us suppose to have a set of $N$ data points $(x_i, y_i), \forall i=1,...,N$ that we want
to use to create our cubic spline.
Let $g(x)$ be the cubic in the interval $x_i \leq x \leq x_{i+1}$ and let $g(x)$ denote the
collection of all the cubics for the entire range of x. Since $g$ is piecewise cubic,
its second derivative $g^{''}$ is piecewise linear. 
For the interval $x_i \leq x \leq x_{i+1}$, we
can then write the equation for the corresponding straight line as
$$
g^{''}_i(x) = g^{''}(x_i)\frac{x-x_{i+1}}{x_{i} - x_{i+1}} + g^{''}(x_{i+1})\frac{x-x_{i}}{x_{i+1} - x_{i}}\;.
$$
Integrating two times and imposing to match the known values at the endpoints 
($y_i = g_i(x_i)$ and $y_{i+1} = g_i(x_{i+1})$) we get:
$$
\begin{split}
g_i(x) = 
    & \frac{g^{''}(x_i)}{6}\left[-\frac{(x - x_{i+1})^3}{\Delta_i} + \Delta_i(x - x_{i+1})\right] + \frac{g^{''}(x_{i+1})}{6}\left[ \frac{(x - x_{i})^3}{\Delta_i} - \Delta_i(x - x_{i})\right] \\
    &+ y_{i+1} \frac{x - x_{i}}{\Delta_i} - y_{i} \frac{x - x_{i+1}}{\Delta_i} \; , 
    \quad \forall x \in [x_i, x_{i+1}] \;, \, \forall i=1,...,N-1 
\end{split}
$$
where $\Delta_i = x_{i+1} - x_i$.

The $g^{''}(x_i)$ are the $N$ unknowns, and applying the continuity of the first derivative condition 
$g^{'}_i(x_i) = g^{'}_{i-1}(x_i) \, , \forall i=2,...,N-1$, we get

$$
\frac{\Delta_{i-1}}{6}g^{''}(x_{i-1}) + 
\frac{\Delta_{i-1} + \Delta_{i}}{3}g^{''}(x_{i}) +
\frac{\Delta_{i}}{6}g^{''}(x_{i+1}) =
\frac{y_{i+1} + y_i}{\Delta_i} - \frac{y_{i} + y_{i-1}}{\Delta_{i-1}}  \; , \quad \forall i = 2,...N-1
$$

We have then $N-2$ equations for $N$ unknowns. The remaining ones are the "end conditions" that one may want to apply.
The most common ones are:
- Free run-out or Natural spline: $g^{''}(x_1)=g^{''}(x_N)=0$
  - most commonly used;
  - this spline is the smoothest interpolant (i.e. $\int_{x_1}^{x_N}g^{''2}(x)\, \mathrm{d}x$ is the smallest possible)
- Parabolic run-out: $g^{''}(x_1)=g^{''}(x_2) \, , \; g^{''}(x_{N-1}) = g^{''}(x_N)$
  - the interpolating polynomials in the first and last intervals are parabolas rather than cubics
- Combination of the first two: $g^{''}(x_1)=\alpha \ g^{''}(x_2) \, , \; g^{''}(x_{N-1}) = \beta \ g^{''}(x_N)$
- Third Derivative continuity: for this look at the appropriate section

Using the coefficients $c_1$, $a_N$, $b_1$, $b_N$, $d_1$ and $d_N$ for this Initial Conditions (IC), we can write the complete set of equation in the matrix form:



$$
\begin{bmatrix}
    b_1 & c_1 & 0 & \cdots & \cdots & 0 \\[5pt]
    \frac{\Delta_{1}}{6}  & \frac{\Delta_{1} + \Delta_{2}}{3} & \frac{\Delta_{2}}{6} & \cdots & \cdots & \vdots \\[5pt]
    0 & \frac{\Delta_{2}}{6}  & \frac{\Delta_{2} + \Delta_{3}}{3} &  \ddots & \cdots & \vdots \\[5pt]
     \vdots & \dots & \ddots & \ddots & \ddots & \vdots \\[5pt]
    \vdots & \dots & \dots & \frac{\Delta_{N-2}}{6} & \frac{\Delta_{N-2} + \Delta_{N-1}}{3} & \frac{\Delta_{N-1}}{6}\\[5pt]
    0 & \cdots & \cdots & 0 & a_N & b_N
\end{bmatrix}
\begin{bmatrix}
    g^{''}(x_1) \\[5pt]
    g^{''}(x_2) \\[5pt]
    \vdots \\[5pt]
    g^{''}(x_{N-1}) \\[5pt]
    g^{''}(x_N)
\end{bmatrix}=
\begin{bmatrix}
    d_1 \\[5pt]
    \frac{y_{3} + y_2}{\Delta_2} - \frac{y_{2} + y_{1}}{\Delta_{1}} \\[5pt]
    \vdots \\[5pt]
    \frac{y_{N} + y_{N-1}}{\Delta_{N-1}} - \frac{y_{N-1} + y_{N-2}}{\Delta_{N-2}} \\[5pt]
    d_N
\end{bmatrix}
$$


## Apply the TDMA 

The equations are in tridiagonal form and diagonally dominant, so we can easily apply the TriDiagonal Matrix Algorithm (TDMA) to this system

$$
\begin{bmatrix}
    b_1 & c_1 & 0 & \cdots & \cdots & 0 \\[5pt]
    a_2  & b_2 & c_2 & \cdots & \cdots & \vdots \\[10 pt]
    0 & a_3  & b_3 &  \ddots & \cdots & \vdots \\[10pt]
     \vdots & \dots & \ddots & \ddots & \ddots & \vdots \\[10pt]
    \vdots & \dots & \dots & a_{N-1} & b_{N-1} & c_{N-1}\\[10pt]
    0 & \cdots & \cdots & 0 & a_N & b_N
\end{bmatrix}
\begin{bmatrix}
    g^{''}(x_1) \\[5pt]
    g^{''}(x_2) \\[5pt]
    \vdots \\[5pt]
    g^{''}(x_{N-1}) \\[5pt]
    g^{''}(x_N)
\end{bmatrix}=
\begin{bmatrix}
    d_1 \\[5pt]
    d_2 \\[5pt]
    \vdots \\[5pt]
    d_{N-1} \\[5pt]
    d_N
\end{bmatrix}
$$
with the following coefficients:
$$
\begin{split}
a_i&=\begin{cases}
        \dfrac{\Delta_{i-1}}{6} \quad,\; \forall i=2,...,N-1 \\[5pt]
        a_N  \quad \quad , \; i=N
    \end{cases}\\[20pt]
b_i&=\begin{cases}
        b_1  \qquad\qquad\qquad \; , \; i=1 \\[5pt]
        \dfrac{\Delta_{i-1} + \Delta_{i}}{3} \quad,\; \forall i=2,...,N-1 \\[5pt]
        b_N  \qquad\qquad\qquad\; , \; i=N
    \end{cases}\\[30pt]
c_i&=\begin{cases}
        c_1  \quad \quad , \; i=1 \\[5pt]
        \dfrac{\Delta_{i}}{6} \quad\;\;,\; \forall i=2,...,N-1
    \end{cases}\\[30pt]
d_i&=\begin{cases}
        d_1  \qquad\qquad\qquad\qquad\quad , \; i=1 \\[5pt]
        \dfrac{y_{i+1} + y_i}{\Delta_i} - \dfrac{y_{i} + y_{i-1}}{\Delta_{i-1}} \quad,\; \forall i=2,...,N-1 \\[5pt]
        d_N  \qquad\qquad\qquad\qquad\quad , \; i=N
    \end{cases}
\end{split}
$$

and the corresponding IC values:
- Free run-out: 
  $$
  b_1=b_N=1,\\ c_1=a_N=d_1=d_N=0
  $$
- Parabolic run-out: 
  $$
    b_1=b_N=1,\\ c_1=a_N=-1, \\ d_1=d_N=0
  $$
- Combination of first two run-out: 
  $$
    b_1=b_N=1,\\ c_1=-\beta\; , \quad a_N=-\alpha, \\ d_1=d_N=0
  $$
- Third Derivative continuity: 
  $$
    c_1=\dfrac{\Delta_1}{6} \; , \quad 
    a_N=\dfrac{\Delta_{N-1}}{6} \; , \quad
    b_1=-\dfrac{\Delta_1}{6} \; , \quad 
    b_N=-\dfrac{\Delta_{N-1}}{6} \, ,\\[10pt]
    d_1=\Delta_1^2\; \eta_1^{(3)} \; , \quad 
    d_N=-\Delta_{N-1}^2\; \eta_{N-3}^{(3)}  \; ,\\[10pt]
    \eta_i^{(3)} = \dfrac{\eta_{i+1}^{(2)}-\eta_{i}^{(2)}}{x_{i+3} - x_i} \; , \quad 
    \eta_i^{(2)} = \dfrac{\eta_{i+1}-\eta_{i}}{x_{i+2} - x_i} \; , \quad
    \eta_i = \dfrac{y_{i+1} - y_i}{x_{i+} - x_i}
  $$


Then, we can compute the new coefficients $\alpha$, $\beta$, $\gamma$ and $\delta$, that give us a new simplified matrix:

$$
\begin{bmatrix}
    \beta_1 & \gamma_1 & 0  & \cdots & 0 \\[5pt]
    0  & \beta_2 & \gamma_2 & \cdots & \vdots \\[10 pt]
    \vdots & \dots & \ddots & \ddots & \vdots \\[10pt]
    \vdots & \dots & \dots  & \beta_{N-1} &\gamma_{N-1}\\[10pt]
    0 & \cdots & \cdots & 0 & \beta_N
\end{bmatrix}
\begin{bmatrix}
    g^{''}(x_1) \\[5pt]
    g^{''}(x_2) \\[5pt]
    \vdots \\[5pt]
    g^{''}(x_{N-1}) \\[5pt]
    g^{''}(x_N)
\end{bmatrix}=
\begin{bmatrix}
    \delta_1 \\[5pt]
    \delta_2 \\[5pt]
    \vdots \\[5pt]
    \delta_{N-1} \\[5pt]
    \delta_N
\end{bmatrix}
$$

with:

$$
\begin{split}
\alpha_i &=0 \; ,\qquad\forall i=2,...,N \\[5pt]
\beta_i&=\begin{cases}
        b_1  \qquad\qquad\qquad\qquad\qquad\qquad , \; i=1 \\[5pt]
        \dfrac{\Delta_{i-1} + \Delta_{i}}{3}\beta_{i-1} - \dfrac{\Delta_{i-1}}{6}\gamma_{i-1} \quad,\; \forall i=2,...,N-1 \\[5pt]
        b_N \beta_{N-1} - a_N \gamma_{N-1}\qquad\qquad\quad , \; i=N
    \end{cases}\\[30pt]
\gamma_i&=\begin{cases}
        c_1  \qquad \qquad , \; i=1 \\[5pt]
        \dfrac{\Delta_{i}}{6}\beta_{i-1} \quad\;\;,\; \forall i=2,...,N-1
    \end{cases}\\[30pt]
\delta_i&=\begin{cases}
        d_1  \qquad\qquad\qquad\qquad , \; i=1 \\[5pt]
        \zeta_i \beta_{i-1} - \dfrac{\Delta_{i-1}}{6}\delta_{i-1} \quad,\; \forall i=2,...,N-1 \\[5pt]
        d_N \beta_{N-1} - a_N \delta_{N-1}  \quad , \; i=N
    \end{cases}\\[30pt]
\zeta_i &= \dfrac{y_{i+1} + y_i}{\Delta_i} - \dfrac{y_{i} + y_{i-1}}{\Delta_{i-1}}
\end{split}
$$

Even better, we can divide everything for $\beta_i$, and obtain:
$$
\begin{bmatrix}
    1 & \tilde{\gamma}_1 & 0  & \cdots & 0 \\[5pt]
    0  & 1 & \tilde{\gamma}_2 & \cdots & \vdots \\[10 pt]
    \vdots & \dots & \ddots & \ddots & \vdots \\[10pt]
    \vdots & \dots & \dots  & 1 &\tilde{\gamma}_{N-1}\\[10pt]
    0 & \cdots & \cdots & 0 & 1
\end{bmatrix}
\begin{bmatrix}
    g^{''}(x_1) \\[5pt]
    g^{''}(x_2) \\[5pt]
    \vdots \\[5pt]
    g^{''}(x_{N-1}) \\[5pt]
    g^{''}(x_N)
\end{bmatrix}=
\begin{bmatrix}
    \tilde{\delta}_1 \\[5pt]
    \tilde{\delta}_2 \\[5pt]
    \vdots \\[5pt]
    \tilde{\delta}_{N-1} \\[5pt]
    \tilde{\delta}_N
\end{bmatrix}
$$

with

$$
\begin{split}
\tilde{\alpha}_i &=\frac{\alpha_i}{\beta_i}=0 \; ,\qquad\forall i=2,...,N \\[5pt]
\tilde{\beta}_i&= \frac{\beta_i}{\beta_i}= 1\; ,\qquad\forall i=1,...,N \\[5pt]
\tilde{\gamma}_i&=\frac{\gamma_i}{\beta_i}=\begin{cases}
        c_1/b_1  \qquad \qquad , \; i=1 \\[5pt]
        c_i \frac{\beta_{i-1}}{\beta_i} = 
        \frac{c_i \beta_{i-1}}{b_i \beta_{i-1} - \gamma_{i-1}a_i} = \frac{c_i}{b_i - \tilde{\gamma}_{i-1}a_i}= \frac{\Delta_i}{2(\Delta_i + \Delta_{i-1}) - \Delta_{i-1}\tilde{\gamma}_{i-1}}\quad\;\;,\; \forall i=2,...,N-1
    \end{cases}\\[30pt]
\tilde{\delta}_i&=\frac{\delta_i}{\beta_i}=\begin{cases}
        d_1/b_1  \qquad\qquad\qquad\qquad , \; i=1 \\[5pt]
        \frac{d_i \beta_{i-1} - \delta_{i-1}a_i}{b_i \beta_{i-1} - \gamma_{i-1}a_i} = \frac{d_i - \tilde{\delta}_{i-1}a_i}{b_i - \tilde{\gamma}_{i-1}a_i} = \frac{6\zeta_i - \Delta_{i-1} \tilde{\delta}_{i-1} }{2(\Delta_{i-1} + \Delta_i)- \Delta_{i-1}\tilde{\gamma}_{i-1}} \quad,\; \forall i=2,...,N-1 \\[5pt]
        d_N \beta_{N-1} - a_N \delta_{N-1}  \quad , \; i=N
    \end{cases}\\[30pt]
\zeta_i &= \dfrac{y_{i+1} + y_i}{\Delta_i} - \dfrac{y_{i} + y_{i-1}}{\Delta_{i-1}}
\end{split}
$$

The unknowns $g^{''}(x_i)$ are given from:
$$
g^{''}(x_N) = \frac{\delta_N}{\beta_N} = \tilde{\delta}_N\\[10pt]
g^{''}(x_i) = \frac{\delta_i - \gamma_i\, g^{''}(x_{i+1})}{\beta_i} = \tilde{\delta}_i - \tilde{\gamma}_i\, g^{''}(x_{i+1})
$$


## Final spline polynomial coefficients 


The final expression for $g(x)$ is:

$$
g(x) = \sum_{i=1}^N g_i(x) \;,\quad \forall x \in [x_1, x_N]\\[10pt]
g_i(x) = y_i + B_i (x-x_i) + C_i (x-x_i)^2 + D_i (x-x_i)^3 \;, \\[5pt]
\forall x \in [x_i, x_{i+1}] \; , \quad \forall i=1,...,N-1
$$

where:
$$
C_i = \frac{g^{''}(x_i)}{2} = \begin{cases}
    \dfrac{\delta_N}{2\beta_N} = \dfrac{\tilde{\delta}_N}{2} \qquad, i=N \\[10pt]
    \dfrac{\delta_i - \gamma_i\, g^{''}(x_{i+1})}{2\beta_i} = \dfrac{\delta_i - \gamma_i\, C_{i+1}}{2\beta_i} =\dfrac{\tilde{\delta}_i - \tilde{\gamma}_i\, C_{i+1}}{2} \; , \; \forall i = 1,...,N-1
\end{cases}
$$
$$
\begin{split}
B_i &= \frac{y_{i+1}-y_i}{\Delta_i} - \frac{\Delta_i}{6}\left[2 g^{''}(x_i) + g^{''}(x_{i+1}) \right] \\[10pt]
&= \frac{y_{i+1}-y_i}{\Delta_i} - \frac{\Delta_i}{3}\left[2 C_i + C_{i+1} \right] \qquad , \; \forall i = 1,...,N-1
\end{split}
$$
$$
\begin{split}
D_i &= \frac{g^{''}(x_{i+1})-g^{''}(x_i)}{6\Delta_i} \\[10pt]
&= \frac{C_{i+1}-C_i}{3\Delta_i} \qquad , \; \forall i = 1,...,N-1
\end{split}
$$

NOTE: the coefficients $B_i$, $C_i$ and $D_i$ are $3(N-1)$ tuples! We defined $C_N$ just to get rid of $g^{''}(x_N)$ and write $B_i$ and $D_i$ in function of $C_i$ only. However, that $C_N$ has no importance after the recursive computation, and you can get rid of it.