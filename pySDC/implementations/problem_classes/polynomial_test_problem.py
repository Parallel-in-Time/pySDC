import numpy as np

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh


class polynomial_testequation(ptype):
    """
    Dummy problem for tests only! In particular, the `solve_system` function just returns the exact solution instead of
    solving an appropriate system. This class is indented to be used for tests of operations that are exact on polynomials.
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, degree=1, seed=26266):
        """Initialization routine"""

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(init=(1, None, np.dtype('float64')))

        self.rng = np.random.RandomState(seed=seed)
        self.poly = np.polynomial.Polynomial(self.rng.rand(degree))
        self._makeAttributeAndRegister('degree', 'seed', localVars=locals(), readOnly=True)

        # differentiation matrix
        self.A = np.diag(np.arange(self.degree - 1) + 1, k=1)
        self.I = np.eye(self.degree)

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
        f[:] = self.poly.deriv(m=1)(t)
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
        u_init : pySDC.problem.testequation0d.dtype_u
            Initial solution.
        t_init : float
            The initial time.

        Returns
        -------
        me : dtype_u
            The exact solution.
        """
        me = self.dtype_u(self.init)
        me[:] = self.poly(t)
        return me
