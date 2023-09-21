import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.helpers import problem_helper
from pySDC.implementations.datatype_classes.mesh import mesh


# noinspection PyUnusedLocal
class generalized_fisher(ptype):
    r"""
    The following one-dimensional problem is an example of a reaction-diffusion equation with traveling waves, and can
    be seen as a generalized Fisher equation. This class implements a special case of the Kolmogorov-Petrovskii-Piskunov
    problem [1]_

    .. math::
        \frac{\partial u}{\partial t} = \Delta u + \lambda_0^2 u (1 - u^\nu)

    with initial condition

    .. math::
        u(x, 0) = \left[
            1 + \left(2^{\nu / 2} - 1\right) \exp\left(-(\nu / 2)\delta x\right)
        \right]^{2 / \nu}

    for :math:`x \in \mathbb{R}`. For

    .. math::
        \delta = \lambda_1 - \sqrt{\lambda_1^2 - \lambda_0^2},

    .. math::
        \lambda_1 = \frac{\lambda_0}{2} \left[
            \left(1 + \frac{\nu}{2}\right)^{1/2} + \left(1 + \frac{\nu}{2}\right)^{-1/2}
        \right],

    the exact solution is

    .. math::
        u(x, t) = \left(
            1 + \left(2^{\nu / 2} - 1\right) \exp\left(-\frac{\nu}{2}\delta (x + 2 \lambda_1 t)\right)
        \right)^{-2 / n}.

    Spatial discretization is done by centered finite differences.

    Parameters
    ----------
    nvars : int, optional
        Spatial resolution, i.e., number of degrees of freedom in space.
    nu : float, optional
        Problem parameter :math:`\nu`.
    lambda0 : float, optional
        Problem parameter :math:`\lambda_0`.
    newton_maxiter : int, optional
        Maximum number of Newton iterations to solve the nonlinear system.
    newton_tol : float, optional
        Tolerance for Newton to terminate.
    interval : tuple, optional
        Defines the spatial domain.
    stop_at_nan : bool, optional
        Indicates if the nonlinear solver should stop if ``nan`` values arise.

    Attributes
    ----------
    A : sparse matrix (CSC)
        Second-order FD discretization of the 1D Laplace operator.
    dx : float
        Distance between two spatial nodes.

    References
    ----------
    .. [1] Z. Feng. Traveling wave behavior for a generalized fisher equation. Chaos Solitons Fractals 38(2),
        481–488 (2008).
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(
        self, nvars=127, nu=1.0, lambda0=2.0, newton_maxiter=100, newton_tol=1e-12, interval=(-5, 5), stop_at_nan=True
    ):
        """Initialization routine"""

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if (nvars + 1) % 2 != 0:
            raise ProblemError('setup requires nvars = 2^p - 1')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__((nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'nvars',
            'nu',
            'lambda0',
            'newton_maxiter',
            'newton_tol',
            'interval',
            'stop_at_nan',
            localVars=locals(),
            readOnly=True,
        )

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
        Simple Newton solver.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            urrent time (required here for the BC).

        Returns
        -------
        u : dtype_u
            Solution.
        """

        u = self.dtype_u(u0)

        nu = self.nu
        lambda0 = self.lambda0

        # set up boundary values to embed inner points
        lam1 = lambda0 / 2.0 * ((nu / 2.0 + 1) ** 0.5 + (nu / 2.0 + 1) ** (-0.5))
        sig1 = lam1 - np.sqrt(lam1**2 - lambda0**2)
        ul = (1 + (2 ** (nu / 2.0) - 1) * np.exp(-nu / 2.0 * sig1 * (self.interval[0] + 2 * lam1 * t))) ** (-2.0 / nu)
        ur = (1 + (2 ** (nu / 2.0) - 1) * np.exp(-nu / 2.0 * sig1 * (self.interval[1] + 2 * lam1 * t))) ** (-2.0 / nu)

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
        Routine to evaluate the right-hand side of the problem.

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
        # set up boundary values to embed inner points
        lam1 = self.lambda0 / 2.0 * ((self.nu / 2.0 + 1) ** 0.5 + (self.nu / 2.0 + 1) ** (-0.5))
        sig1 = lam1 - np.sqrt(lam1**2 - self.lambda0**2)
        ul = (1 + (2 ** (self.nu / 2.0) - 1) * np.exp(-self.nu / 2.0 * sig1 * (self.interval[0] + 2 * lam1 * t))) ** (
            -2 / self.nu
        )
        ur = (1 + (2 ** (self.nu / 2.0) - 1) * np.exp(-self.nu / 2.0 * sig1 * (self.interval[1] + 2 * lam1 * t))) ** (
            -2 / self.nu
        )

        uext = np.concatenate(([ul], u, [ur]))

        f = self.dtype_f(self.init)
        f[:] = self.A.dot(uext)[1:-1] + self.lambda0**2 * u * (1 - u**self.nu)
        return f

    def u_exact(self, t):
        r"""
        Routine to compute the exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            Exact solution.
        """

        me = self.dtype_u(self.init)
        xvalues = np.array([(i + 1 - (self.nvars + 1) / 2) * self.dx for i in range(self.nvars)])

        lam1 = self.lambda0 / 2.0 * ((self.nu / 2.0 + 1) ** 0.5 + (self.nu / 2.0 + 1) ** (-0.5))
        sig1 = lam1 - np.sqrt(lam1**2 - self.lambda0**2)
        me[:] = (1 + (2 ** (self.nu / 2.0) - 1) * np.exp(-self.nu / 2.0 * sig1 * (xvalues + 2 * lam1 * t))) ** (
            -2.0 / self.nu
        )
        return me
