import numpy as np

from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh


# noinspection PyUnusedLocal
class nonlinear_ODE_1(ptype):
    r"""
    This class implements a simple nonlinear ODE with a singularity in the derivative, taken from
    https://www.osti.gov/servlets/purl/6111421 (Problem E-4). For :math:`0 \leq t \leq 5`, the problem is
    given by

    .. math::
        \frac{du(t)}{dt} = \sqrt(1 - u(t))

    with initial condition :math:`u(0) = 0`. The exact solution is

    .. math::
        u(t) = t - \frac{t^2}{4}.

    Parameters
    ----------
    u0 : float, optional
        Initial condition.
    newton_maxiter : float, optional
        Maximum number of iterations for Newton's method.
    newton_tol : float, optional
        Tolerance for Newton's method to terminate.
    stop_at_nan : bool, optional
        Indicates that Newton solver has to stop if nan values arise.
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, u0=0.0, newton_maxiter=200, newton_tol=5e-11, stop_at_nan=True):
        nvars = 1
        super().__init__((nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'u0', 'newton_maxiter', 'newton_tol', 'stop_at_nan', localVars=locals(), readOnly=True
        )

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            The exact solution.
        """

        me = self.dtype_u(self.init)
        me[:] = t - t**2 / 4
        return me

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed (not used here).

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem (one component).
        """

        f = self.dtype_f(self.init)
        f[:] = np.sqrt(1 - u)
        return f

    def solve_system(self, rhs, dt, u0, t):
        """
        Simple Newton solver for the nonlinear equation

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        dt : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        u : dtype_u
            The solution as mesh.
        """
        # create new mesh object from u0 and set initial values for iteration
        u = self.dtype_u(u0)

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # form the function g with g(u) = 0
            g = u - dt * np.sqrt(1 - u) - rhs

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol or np.isnan(res):
                break

            # assemble dg/du
            dg = 1 - (-dt) / (2 * np.sqrt(1 - u))
            # newton update: u1 = u0 - g/dg
            u -= 1.0 / dg * g

            # increase iteration count
            n += 1

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.newton_maxiter:
            raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

        return u
