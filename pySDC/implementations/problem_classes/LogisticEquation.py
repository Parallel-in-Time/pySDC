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

    with :math:`\lambda` a given real coefficient. Its analytical solution is :

    .. math::
        u(t) = u(0) \frac{e^{\lambda t}}{1-u(0)+u(0)e^{\lambda t}}

    """
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, u0, newton_maxiter, newton_tol, direct, lam=1, stop_at_nan=True):
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
        Exact solution

        Args:
            t (float): current time
        Returns:
            dtype_u: mesh type containing the values
        """

        me = self.dtype_u(self.init)
        me[:] = self.u0 * np.exp(self.lam * t) / (1 - self.u0 + self.u0 * np.exp(self.lam * t))
        return me

    def eval_f(self, u, t):
        """
        Routine to compute the RHS

        Args:
            u (dtype_u): the current values
            t (float): current time (not used here)
        Returns:
            dtype_f: RHS, 1 component
        """

        f = self.dtype_f(self.init)
        f[:] = self.lam * u * (1 - u)
        return f

    def solve_system(self, rhs, dt, u0, t):
        """
        Simple Newton solver for the nonlinear equation

        Args:
            rhs (dtype_f): right-hand side for the nonlinear system
            dt (float): abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution u
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
