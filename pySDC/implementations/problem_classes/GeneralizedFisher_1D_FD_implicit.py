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

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'nu', 'lambda0', 'newton_maxiter', 'newton_tol', 'interval']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if (problem_params['nvars'] + 1) % 2 != 0:
            raise ProblemError('setup requires nvars = 2^p - 1')

        if 'stop_at_nan' not in problem_params:
            problem_params['stop_at_nan'] = True

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(generalized_fisher, self).__init__(
            (problem_params['nvars'], None, np.dtype('float64')), dtype_u, dtype_f, problem_params
        )

        # compute dx and get discretization matrix A
        self.dx = (self.params.interval[1] - self.params.interval[0]) / (self.params.nvars + 1)
        self.A = problem_helper.get_finite_difference_matrix(
            derivative=2,
            order=2,
            type='center',
            dx=self.dx,
            size=self.params.nvars + 2,
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

        nu = self.params.nu
        lambda0 = self.params.lambda0

        # set up boundary values to embed inner points
        lam1 = lambda0 / 2.0 * ((nu / 2.0 + 1) ** 0.5 + (nu / 2.0 + 1) ** (-0.5))
        sig1 = lam1 - np.sqrt(lam1**2 - lambda0**2)
        ul = (1 + (2 ** (nu / 2.0) - 1) * np.exp(-nu / 2.0 * sig1 * (self.params.interval[0] + 2 * lam1 * t))) ** (
            -2.0 / nu
        )
        ur = (1 + (2 ** (nu / 2.0) - 1) * np.exp(-nu / 2.0 * sig1 * (self.params.interval[1] + 2 * lam1 * t))) ** (
            -2.0 / nu
        )

        # start newton iteration
        n = 0
        res = 99
        while n < self.params.newton_maxiter:

            # form the function g with g(u) = 0
            uext = np.concatenate(([ul], u, [ur]))
            g = u - factor * (self.A.dot(uext)[1:-1] + lambda0**2 * u * (1 - u**nu)) - rhs

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.params.newton_tol:
                break

            # assemble dg
            dg = sp.eye(self.params.nvars) - factor * (
                self.A[1:-1, 1:-1] + sp.diags(lambda0**2 - lambda0**2 * (nu + 1) * u**nu, offsets=0)
            )

            # newton update: u1 = u0 - g/dg
            u -= spsolve(dg, g)

            # increase iteration count
            n += 1

        if np.isnan(res) and self.params.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.params.newton_maxiter:
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
        lam1 = self.params.lambda0 / 2.0 * ((self.params.nu / 2.0 + 1) ** 0.5 + (self.params.nu / 2.0 + 1) ** (-0.5))
        sig1 = lam1 - np.sqrt(lam1**2 - self.params.lambda0**2)
        ul = (
            1
            + (2 ** (self.params.nu / 2.0) - 1)
            * np.exp(-self.params.nu / 2.0 * sig1 * (self.params.interval[0] + 2 * lam1 * t))
        ) ** (-2 / self.params.nu)
        ur = (
            1
            + (2 ** (self.params.nu / 2.0) - 1)
            * np.exp(-self.params.nu / 2.0 * sig1 * (self.params.interval[1] + 2 * lam1 * t))
        ) ** (-2 / self.params.nu)

        uext = np.concatenate(([ul], u, [ur]))

        f = self.dtype_f(self.init)
        f[:] = self.A.dot(uext)[1:-1] + self.params.lambda0**2 * u * (1 - u**self.params.nu)
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
        xvalues = np.array([(i + 1 - (self.params.nvars + 1) / 2) * self.dx for i in range(self.params.nvars)])

        lam1 = self.params.lambda0 / 2.0 * ((self.params.nu / 2.0 + 1) ** 0.5 + (self.params.nu / 2.0 + 1) ** (-0.5))
        sig1 = lam1 - np.sqrt(lam1**2 - self.params.lambda0**2)
        me[:] = (
            1 + (2 ** (self.params.nu / 2.0) - 1) * np.exp(-self.params.nu / 2.0 * sig1 * (xvalues + 2 * lam1 * t))
        ) ** (-2.0 / self.params.nu)
        return me
