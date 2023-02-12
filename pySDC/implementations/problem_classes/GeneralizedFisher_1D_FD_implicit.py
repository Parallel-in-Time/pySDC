import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.helpers import problem_helper
from pySDC.implementations.datatype_classes.mesh import mesh


# noinspection PyUnusedLocal
class generalized_fisher(ptype):
    """
    Example implementing the generalized Fisher's equation in 1D with finite differences

    Attributes:
        A: second-order FD discretization of the 1D laplace operator
        dx: distance between two spatial nodes
    """
    dtype_u = mesh
    dtype_f = mesh

    def __init__(
            self, nvars, nu, lambda0, newton_maxiter, newton_tol, interval,
            stop_at_nan=True):
        """Initialization routine"""

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if (nvars + 1) % 2 != 0:
            raise ProblemError('setup requires nvars = 2^p - 1')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__((nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'nvars', 'nu', 'lambda0', 'newton_maxiter', 'newton_tol', 'interval',
            'stop_at_nan', localVars=locals(), readOnly=True)

        # compute dx and get discretization matrix A
        self.dx = (self.interval[1] - self.interval[0]) / (self.nvars + 1)
        self.A = problem_helper.get_finite_difference_matrix(
            derivative=2,
            order=2,
            stencil_type='center',
            dx=self.dx,
            size=self.nvars + 2,
            dim=1,
            bc='dirichlet-zero',
        )

    # noinspection PyTypeChecker
    def solve_system(self, rhs, factor, u0, t):
        """
        Simple Newton solver

        Args:
            rhs (dtype_f): right-hand side for the nonlinear system
            factor (float): abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (required here for the BC)

        Returns:
            dtype_u: solution u
        """

        u = self.dtype_u(u0)

        nu = self.nu
        lambda0 = self.lambda0

        # set up boundary values to embed inner points
        lam1 = lambda0 / 2.0 * ((nu / 2.0 + 1) ** 0.5 + (nu / 2.0 + 1) ** (-0.5))
        sig1 = lam1 - np.sqrt(lam1**2 - lambda0**2)
        ul = (1 + (2 ** (nu / 2.0) - 1) * np.exp(-nu / 2.0 * sig1 * (self.interval[0] + 2 * lam1 * t))) ** (
            -2.0 / nu
        )
        ur = (1 + (2 ** (nu / 2.0) - 1) * np.exp(-nu / 2.0 * sig1 * (self.interval[1] + 2 * lam1 * t))) ** (
            -2.0 / nu
        )

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # form the function g with g(u) = 0
            uext = np.concatenate(([ul], u, [ur]))
            g = u - factor * (self.A.dot(uext)[1:-1] + lambda0**2 * u * (1 - u**nu)) - rhs

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.newton_tol:
                break

            # assemble dg
            dg = sp.eye(self.nvars) - factor * (
                self.A[1:-1, 1:-1] + sp.diags(lambda0**2 - lambda0**2 * (nu + 1) * u**nu, offsets=0)
            )

            # newton update: u1 = u0 - g/dg
            u -= spsolve(dg, g)

            # increase iteration count
            n += 1

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.newton_maxiter:
            self.logger.warning('Newton did not converge after %i iterations, error is %s' % (n, res))

        return u

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """
        # set up boundary values to embed inner points
        lam1 = self.lambda0 / 2.0 * ((self.nu / 2.0 + 1) ** 0.5 + (self.nu / 2.0 + 1) ** (-0.5))
        sig1 = lam1 - np.sqrt(lam1**2 - self.lambda0**2)
        ul = (
            1
            + (2 ** (self.nu / 2.0) - 1)
            * np.exp(-self.nu / 2.0 * sig1 * (self.interval[0] + 2 * lam1 * t))
        ) ** (-2 / self.nu)
        ur = (
            1
            + (2 ** (self.nu / 2.0) - 1)
            * np.exp(-self.nu / 2.0 * sig1 * (self.interval[1] + 2 * lam1 * t))
        ) ** (-2 / self.nu)

        uext = np.concatenate(([ul], u, [ur]))

        f = self.dtype_f(self.init)
        f[:] = self.A.dot(uext)[1:-1] + self.lambda0**2 * u * (1 - u**self.nu)
        return f

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        me = self.dtype_u(self.init)
        xvalues = np.array([(i + 1 - (self.nvars + 1) / 2) * self.dx for i in range(self.nvars)])

        lam1 = self.lambda0 / 2.0 * ((self.nu / 2.0 + 1) ** 0.5 + (self.nu / 2.0 + 1) ** (-0.5))
        sig1 = lam1 - np.sqrt(lam1**2 - self.lambda0**2)
        me[:] = (
            1 + (2 ** (self.nu / 2.0) - 1) * np.exp(-self.nu / 2.0 * sig1 * (xvalues + 2 * lam1 * t))
        ) ** (-2.0 / self.nu)
        return me
