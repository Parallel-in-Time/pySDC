import numpy as np

from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh


# noinspection PyUnusedLocal
class logistics_equation(ptype):
    r"""
    Problem implementing a specific form of the Logistic Differential Equation

    .. math::
        \frac{du}{dt} = \lambda u(t)(1-u(t))

    with :math:`\lambda` a given real coefficient. Its analytical solution is
    given by

    .. math::
        u(t) = u(0) \frac{e^{\lambda t}}{1-u(0)+u(0)e^{\lambda t}}

    Parameters
    ----------
    u0 : float, optional
        Initial condition.
    newton_maxiter : int, optional
        Maximum number of iterations for Newton's method.
    newton_tol : float, optional
        Tolerance for Newton's method to terminate.
    direct : bool, optional
        Indicates if the problem should be solved directly or not. If False, it will be approximated by Newton.
    lam : float, optional
        Problem parameter :math:`\lambda`.
    stop_at_nan : bool, optional
        Indicates if the Newton solver stops when nan values arise.
    """
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, u0=0.5, newton_maxiter=100, newton_tol=1e-12, direct=True, lam=1, stop_at_nan=True):
        nvars = 1

        super().__init__((nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'u0',
            'lam',
            'newton_maxiter',
            'newton_tol',
            'direct',
            'nvars',
            'stop_at_nan',
            localVars=locals(),
            readOnly=True,
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
        me[:] = self.u0 * np.exp(self.lam * t) / (1 - self.u0 + self.u0 * np.exp(self.lam * t))
        return me

    def eval_f(self, u, t):
        """
        Routine to compute the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem (contains one component).
        """

        f = self.dtype_f(self.init)
        f[:] = self.lam * u * (1 - u)
        return f

    def solve_system(self, rhs, dt, u0, t):
        """
        Simple Newton solver for the nonlinear equation.

        Parameters
        ----------
        rhs : dtype_f)
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

        if self.direct:
            d = (1 - dt * self.lam) ** 2 + 4 * dt * self.lam * rhs
            u = (-(1 - dt * self.lam) + np.sqrt(d)) / (2 * dt * self.lam)
            return u

        else:
            # start newton iteration
            n = 0
            res = 99
            while n < self.newton_maxiter:
                # form the function g with g(u) = 0
                g = u - dt * self.lam * u * (1 - u) - rhs

                # if g is close to 0, then we are done
                res = np.linalg.norm(g, np.inf)
                if res < self.newton_tol or np.isnan(res):
                    break

                # assemble dg/du
                dg = 1 - dt * self.lam * (1 - 2 * u)
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
