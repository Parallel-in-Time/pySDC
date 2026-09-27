import numpy as np

from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class polynomial_testequation(Problem):
    """
    Scalar ODE whose exact solution is a random polynomial in time, to test operations exact on polynomials.

    Dummy problem for tests only! In particular, the `solve_system` function just returns the exact solution instead of
    solving an appropriate system. This class is indented to be used for tests of operations that are exact on polynomials.

    Parameters
    ----------
    degree : int, optional
        Number of coefficients of the polynomial, i.e., the polynomial has degree ``degree - 1``.
    seed : int, optional
        Seed for ``np.random.RandomState``, which draws the coefficients uniformly from :math:`[0, 1)`.
    useGPU : bool, optional
        Use ``CuPy`` for the solution. This switches the data types of the class, not only of this instance.
    """

    dtype_u = mesh
    dtype_f = mesh
    xp = np

    def __init__(self, degree=1, seed=26266, useGPU=False):
        """Initialization routine"""

        if useGPU:
            import cupy as cp
            from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh, imex_cupy_mesh

            # on the instance, since setting them on the class would switch every later instance to the GPU
            self.xp = cp
            self.dtype_u = cupy_mesh
            self.dtype_f = imex_cupy_mesh if self.dtype_f is imex_mesh else cupy_mesh

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(init=(1, None, np.dtype('float64')))

        self.rng = np.random.RandomState(seed=seed)
        self.poly = np.polynomial.Polynomial(self.rng.rand(degree))
        self._makeAttributeAndRegister('degree', 'seed', localVars=locals(), readOnly=True)

    def eval_f(self, u, t):
        """
        Derivative of the polynomial.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """

        f = self.dtype_f(self.init)
        f[:] = self.xp.array(self.poly.deriv(m=1)(t))
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Just return the exact solution...

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        return self.u_exact(t)

    def u_exact(self, t, **kwargs):
        """
        Evaluate the polynomial.

        Parameters
        ----------
        t : float
            Time of the exact solution.
        **kwargs
            Takes ``u_init`` and ``t_init`` of the generic interface, which are ignored since the polynomial is known
            everywhere.

        Returns
        -------
        me : dtype_u
            The exact solution.
        """
        me = self.dtype_u(self.init)
        me[:] = self.xp.array(self.poly(t))
        return me


class polynomial_testequation_IMEX(polynomial_testequation):
    """
    IMEX version of the polynomial test problem that assigns half the derivative to the implicit part and the other half to the explicit part.
    Keep in mind that you still cannot Really perform any solves.
    """

    dtype_f = imex_mesh

    def eval_f(self, u, t):
        """
        Derivative of the polynomial.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """

        f = self.dtype_f(self.init)
        derivative = self.xp.array(self.poly.deriv(m=1)(t))
        f.impl[:] = derivative / 2
        f.expl[:] = derivative / 2
        return f
