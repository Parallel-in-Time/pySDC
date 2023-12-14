# Numerical experiment scripts for the parallel SDC paper

Common utility scripts :

- [`utils.py`](./utils.py) : utility functions to get numerical solutions with SDC, or reference solutions with scipy (exact)

> :mega: Each script can generate pdf plots with name starting with the name of the script, stored in the [`fig`](./fig/) folder

## Van der Pol

- [`vanderpol_setup.py`](./vanderpol_setup.py) : determine period for different `mu` values, and plot scaled exact solution on one period
- [`vanderpol_accuracy.py`](./vanderpol_accuracy.py) : investigate accuracy of diagonal SDC variants on the different Van der Pol problems

## Lorenz

- [`lorenz_setup.py`](./lorenz_setup.py) : plot numerical solution of Lorenz, and determine numerically the final time of a given number of revolution periods
- [`lorenz_accuracy.py`](./lorenz_accuracy.py) : investigate accuracy of diagonal SDC variants on the Lorenz system on a given simulation time (given number of revolution periods)

## Prothero-Robinson

From [[Houwen & Sommeijer 1991]](#houwen1991iterated), that took it from an other reference of the two guys from which the problem is named :

$$
\frac{du}{dt} = \frac{u-g(t)}{\epsilon} + \frac{dg}{dt},
\quad u(0) = g(0).
$$

The analytical solution is $u_1(t)=\exp(-2t)$ 
and $u_2(t)=\exp(-t)$ whatever $\epsilon$ value, and the smaller 
the $\epsilon$, the stiffer the problem. 

The exact solution of this problem is $g(t)$, 
whatever the value of $\epsilon$. The smaller the later parameter, 
the stiffer the problem is (numerically speaking).

There is also a non-linear version of this problem :

$$
\frac{du}{dt} = \frac{u^3-g(t)^3}{\epsilon} + \frac{dg}{dt},
\quad u(0) = g(0).
$$

that has exactly the same analytical solution, and similar behavior 
considering the $\epsilon$ parameter.

For both linear and non-linear problems, 
we set $g(t)=\cost(t)$ 
and $T=2\pi$.

- [`protheroRobinson_setup.py`](./protheroRobinson_setup.py) : numerical solution for linear and non-linear case, for a given epsilon value
- [`protheroRobinson_accuracy.py`](./protheroRobinson_accuracy.py) : investigate accuracy of diagonal SDC variants

## Kaps

From [[Houwen & Sommeijer 1991]](#houwen1991iterated), that took it from an other reference of some guys named Kaps investigating Rosenbrock methods :

$$
\frac{d{\bf u}}{dt} = \begin{pmatrix}
    -(2+\epsilon^{-1})u_1 + \epsilon^{-1} (u_2)^2 \\
    u_1 - u_2(1+u_2)
\end{pmatrix}, 
\quad {\bf u}(0) = \begin{pmatrix}
1 \\ 1
\end{pmatrix}, \quad T=1.
$$

- [`kaps_setup.py`](./kaps_setup.py) : numerical solution for a given epsilon value
- [`kaps_accuracy.py`](./kaps_accuracy.py) : investigate accuracy of diagonal SDC variants

## References

<a id="houwen1991iterated">[Houwen & Sommeijer 1991]</a> _Van der Houwen, P. J., & Sommeijer, B. P. (1991). [Iterated Runge–Kutta methods on parallel computers](https://epubs.siam.org/doi/pdf/10.1137/0912054). SIAM Journal on Scientific and Statistical Computing, 12(5), 1000-1028._
