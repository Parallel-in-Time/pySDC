import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype
from pySDC.helpers import problem_helper
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh, comp2_mesh


class allencahn_front_fullyimplicit(ptype):
    r"""
    Example implementing the one-dimensional Allen-Cahn equation with driving force using inhomogeneous Dirichlet
    boundary conditions

    .. math::
        \frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2} - \frac{2}{\varepsilon^2} u (1 - u) (1 - 2u)
            - 6 d_w u (1 - u)

    for :math:`u \in [0, 1]`. The second order spatial derivative is approximated using centered finite differences. The
    exact solution is given by

    .. math::
        u(x, t)= 0.5 \left(1 + \tanh\left(\frac{x - vt}{\sqrt{2}\varepsilon}\right)\right)

    with :math:`v = 3 \sqrt{2} \varepsilon d_w`. For time-stepping, this problem is implemented to be treated
    *fully-implicit* using Newton to solve the nonlinear system.

    Parameters
    ----------
    nvars : int
        Number of unknowns in the problem.
    dw : float
        Driving force.
    eps : float
        Scaling parameter :math:`\varepsilon`.
    newton_maxiter : int
        Maximum number of iterations for Newton's method.
    newton_tol : float
        Tolerance for Newton's method to terminate.
    interval : list
        Interval of spatial domain.
    stop_at_nan : bool, optional
        Indicates that the Newton solver should stop if ``nan`` values arise.

    Attributes
    ----------
    A : scipy.diags
        Second-order FD discretization of the 1D laplace operator.
    dx : float
        Distance between two spatial nodes.
    xvalues : np.1darray
        Spatial grid values.
    uext : dtype_u
        Contains additionally the external values of the boundary.
    newton_itercount : int
        Counter for iterations in Newton solver.
    lin_itercount : int
        Counter for iterations in linear solver.
    newton_ncalls : int
        Number of calls of Newton solver.
    lin_ncalls : int
        Number of calls of linear solver.
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(
        self,
        nvars=127,
        dw=-0.04,
        eps=0.04,
        newton_maxiter=100,
        newton_tol=1e-12,
        interval=(-0.5, 0.5),
        stop_at_nan=True,
    ):
        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if (nvars + 1) % 2:
            raise ProblemError('setup requires nvars = 2^p - 1')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__((nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'nvars',
            'dw',
            'eps',
            'newton_maxiter',
            'newton_tol',
            'interval',
            'stop_at_nan',
            localVars=locals(),
            readOnly=True,
        )

        # compute dx and get discretization matrix A
        self.dx = (self.interval[1] - self.interval[0]) / (self.nvars + 1)
        self.xvalues = np.array([(i + 1 - (self.nvars + 1) / 2) * self.dx for i in range(self.nvars)])

        self.A = problem_helper.get_finite_difference_matrix(
            derivative=2,
            order=2,
            type='center',
            dx=self.dx,
            size=self.nvars + 2,
            dim=1,
            bc='dirichlet-zero',
        )
        self.uext = self.dtype_u((self.init[0] + 2, self.init[1], self.init[2]), val=0.0)

        self.newton_itercount = 0
        self.lin_itercount = 0
        self.newton_ncalls = 0
        self.lin_ncalls = 0

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
            Current time (required here for the BC).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        u = self.dtype_u(u0)
        eps2 = self.eps**2
        dw = self.dw

        Id = sp.eye(self.nvars)

        v = 3.0 * np.sqrt(2) * self.eps * self.dw
        self.uext[0] = 0.5 * (1 + np.tanh((self.interval[0] - v * t) / (np.sqrt(2) * self.eps)))
        self.uext[-1] = 0.5 * (1 + np.tanh((self.interval[1] - v * t) / (np.sqrt(2) * self.eps)))

        A = self.A[1:-1, 1:-1]
        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # print(n)
            # # form the function g(u), such that the solution to the nonlinear problem is a root of g
            self.uext[1:-1] = u[:]
            g = (
                u
                - rhs
                - factor
                * (
                    self.A.dot(self.uext)[1:-1]
                    - 2.0 / eps2 * u * (1.0 - u) * (1.0 - 2.0 * u)
                    - 6.0 * dw * u * (1.0 - u)
                )
            )

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (
                A
                - 2.0
                / eps2
                * sp.diags((1.0 - u) * (1.0 - 2.0 * u) - u * ((1.0 - 2.0 * u) + 2.0 * (1.0 - u)), offsets=0)
                - 6.0 * dw * sp.diags((1.0 - u) - u, offsets=0)
            )

            # newton update: u1 = u0 - g/dg
            u -= spsolve(dg, g)
            # u -= gmres(dg, g, x0=z, tol=self.lin_tol)[0]
            # increase iteration count
            n += 1

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.newton_maxiter:
            self.logger.warning('Newton did not converge after %i iterations, error is %s' % (n, res))

        self.newton_ncalls += 1
        self.newton_itercount += n

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me

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
        v = 3.0 * np.sqrt(2) * self.eps * self.dw
        self.uext[0] = 0.5 * (1 + np.tanh((self.interval[0] - v * t) / (np.sqrt(2) * self.eps)))
        self.uext[-1] = 0.5 * (1 + np.tanh((self.interval[1] - v * t) / (np.sqrt(2) * self.eps)))

        self.uext[1:-1] = u[:]

        f = self.dtype_f(self.init)
        f[:] = (
            self.A.dot(self.uext)[1:-1]
            - 2.0 / self.eps**2 * u * (1.0 - u) * (1.0 - 2 * u)
            - 6.0 * self.dw * u * (1.0 - u)
        )
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
            The exact solution.
        """

        v = 3.0 * np.sqrt(2) * self.eps * self.dw
        me = self.dtype_u(self.init, val=0.0)
        me[:] = 0.5 * (1 + np.tanh((self.xvalues - v * t) / (np.sqrt(2) * self.eps)))
        return me


class allencahn_front_semiimplicit(allencahn_front_fullyimplicit):
    r"""
    This class implements the one-dimensional Allen-Cahn equation with driving force using inhomogeneous Dirichlet
    boundary conditions

    .. math::
        \frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2} - \frac{2}{\varepsilon^2} u (1 - u) (1 - 2u)
            - 6 d_w u (1 - u)

    for :math:`u \in [0, 1]`. Centered finite differences are used for discretization of the second order spatial derivative.
    The exact solution is given by

    .. math::
        u(x, t) = 0.5 \left(1 + \tanh\left(\frac{x - vt}{\sqrt{2}\varepsilon}\right)\right)

    with :math:`v = 3 \sqrt{2} \varepsilon d_w`. For time-stepping, this problem will be treated in a
    *semi-implicit* way, i.e., the Laplacian is treated implicitly, and the rest of the right-hand side will be handled
    explicitly.
    """

    dtype_f = imex_mesh

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
        v = 3.0 * np.sqrt(2) * self.eps * self.dw
        self.uext[0] = 0.5 * (1 + np.tanh((self.interval[0] - v * t) / (np.sqrt(2) * self.eps)))
        self.uext[-1] = 0.5 * (1 + np.tanh((self.interval[1] - v * t) / (np.sqrt(2) * self.eps)))

        self.uext[1:-1] = u[:]

        f = self.dtype_f(self.init)
        f.impl[:] = self.A.dot(self.uext)[1:-1]
        f.expl[:] = -2.0 / self.eps**2 * u * (1.0 - u) * (1.0 - 2 * u) - 6.0 * self.dw * u * (1.0 - u)
        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Simple linear solver for :math:`(I-factor\cdot A)\vec{u}=\vec{rhs}`.

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

        me = self.dtype_u(self.init)
        self.uext[0] = 0.0
        self.uext[-1] = 0.0
        self.uext[1:-1] = rhs[:]
        me[:] = spsolve(sp.eye(self.nvars + 2, format='csc') - factor * self.A, self.uext)[1:-1]
        return me


class allencahn_front_finel(allencahn_front_fullyimplicit):
    r"""
    This class implements the one-dimensional Allen-Cahn equation with driving force using inhomogeneous Dirichlet
    boundary conditions

    .. math::
        \frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2} - \frac{2}{\varepsilon^2} u (1 - u) (1 - 2u)
            - 6 d_w u (1 - u)

    for :math:`u \in [0, 1]`. Centered finite differences are used for discretization of the Laplacian.
    The exact solution is given by

    .. math::
        u(x, t) = 0.5 \left(1 + \tanh\left(\frac{x - vt}{\sqrt{2}\varepsilon}\right)\right)

    with :math:`v = 3 \sqrt{2} \varepsilon d_w`.

    Let :math:`A` denote the finite difference matrix to discretize :math:`\frac{\partial^2 u}{\partial x^2}`. Here,
    *Finel's trick* is used. Let

    .. math::
        a = \tanh\left(\frac{\Delta x}{\sqrt{2}\varepsilon}\right)^2,

    then, the right-hand side of the problem can be written as

    .. math::
        \frac{\partial u}{\partial t} = A u - \frac{1}{\Delta x^2} \left[
                \frac{1 - a}{1 - a (2u - 1)^2} - 1
            \right] (2u - 1).

    For time-stepping, this problem will be treated in a *fully-implicit* way. The nonlinear system is solved using Newton.
    """

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
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        u = self.dtype_u(u0)
        dw = self.dw
        a2 = np.tanh(self.dx / (np.sqrt(2) * self.eps)) ** 2

        Id = sp.eye(self.nvars)

        v = 3.0 * np.sqrt(2) * self.eps * self.dw
        self.uext[0] = 0.5 * (1 + np.tanh((self.interval[0] - v * t) / (np.sqrt(2) * self.eps)))
        self.uext[-1] = 0.5 * (1 + np.tanh((self.interval[1] - v * t) / (np.sqrt(2) * self.eps)))

        A = self.A[1:-1, 1:-1]
        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # print(n)
            # form the function g(u), such that the solution to the nonlinear problem is a root of g
            self.uext[1:-1] = u[:]
            gprim = 1.0 / self.dx**2 * ((1.0 - a2) / (1.0 - a2 * (2.0 * u - 1.0) ** 2) - 1.0) * (2.0 * u - 1.0)
            g = u - rhs - factor * (self.A.dot(self.uext)[1:-1] - 1.0 * gprim - 6.0 * dw * u * (1.0 - u))

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.newton_tol:
                break

            # assemble dg
            dgprim = (
                1.0
                / self.dx**2
                * (
                    2.0 * ((1.0 - a2) / (1.0 - a2 * (2.0 * u - 1.0) ** 2) - 1.0)
                    + (2.0 * u - 1) ** 2 * (1.0 - a2) * 4 * a2 / (1.0 - a2 * (2.0 * u - 1.0) ** 2) ** 2
                )
            )

            dg = Id - factor * (A - 1.0 * sp.diags(dgprim, offsets=0) - 6.0 * dw * sp.diags((1.0 - u) - u, offsets=0))

            # newton update: u1 = u0 - g/dg
            u -= spsolve(dg, g)
            # For some reason, doing cg or gmres does not work so well here...
            # u -= cg(dg, g, x0=z, tol=self.lin_tol)[0]
            # increase iteration count
            n += 1

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.newton_maxiter:
            self.logger.warning('Newton did not converge after %i iterations, error is %s' % (n, res))

        self.newton_ncalls += 1
        self.newton_itercount += n

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me

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
        v = 3.0 * np.sqrt(2) * self.eps * self.dw
        self.uext[0] = 0.5 * (1 + np.tanh((self.interval[0] - v * t) / (np.sqrt(2) * self.eps)))
        self.uext[-1] = 0.5 * (1 + np.tanh((self.interval[1] - v * t) / (np.sqrt(2) * self.eps)))

        self.uext[1:-1] = u[:]

        a2 = np.tanh(self.dx / (np.sqrt(2) * self.eps)) ** 2
        gprim = 1.0 / self.dx**2 * ((1.0 - a2) / (1.0 - a2 * (2.0 * u - 1.0) ** 2) - 1) * (2.0 * u - 1.0)
        f = self.dtype_f(self.init)
        f[:] = self.A.dot(self.uext)[1:-1] - 1.0 * gprim - 6.0 * self.dw * u * (1.0 - u)
        return f


class allencahn_periodic_fullyimplicit(ptype):
    r"""
    Example implementing the one-dimensional Allen-Cahn equation with driving force and periodic boundary conditions

    .. math::
        \frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2} - \frac{2}{\varepsilon^2} u (1 - u) (1 - 2u)
            - 6 d_w u (1 - u)

    for :math:`u \in [0, 1]`. Centered finite differences are used for discretization of the Laplacian.
    The exact solution is

    .. math::
        u(x, t) = 0.5 \left(1 + \tanh\left(\frac{r - |x| - vt}{\sqrt{2}\varepsilon}\right)\right)

    with :math:`v = 3 \sqrt{2} \varepsilon d_w` and radius :math:`r` of the circles. For time-stepping, the problem is treated
    *fully-implicitly*, i.e., the nonlinear system is solved by Newton.

    Parameters
    ----------
    nvars : int
        Number of unknowns in the problem.
    dw : float
        Driving force.
    eps : float
        Scaling parameter :math:`\varepsilon`.
    newton_maxiter : int
        Maximum number of iterations for Newton's method.
    newton_tol : float
        Tolerance for Newton's method to terminate.
    interval : list
        Interval of spatial domain.
    radius : float
        Radius of the circles.
    stop_at_nan : bool, optional
        Indicates that the Newton solver should stop if nan values arise.

    Attributes
    ----------
    A : scipy.diags
        Second-order FD discretization of the 1D laplace operator.
    dx : float
        Distance between two spatial nodes.
    xvalues : np.1darray
        Spatial grid points.
    newton_itercount : int
        Number of iterations for Newton solver.
    lin_itercount : int
        Number of iterations for linear solver.
    newton_ncalls : int
        Number of calls of Newton solver.
    lin_ncalls : int
        Number of calls of linear solver.
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(
        self,
        nvars=128,
        dw=-0.04,
        eps=0.04,
        newton_maxiter=100,
        newton_tol=1e-12,
        interval=(-0.5, 0.5),
        radius=0.25,
        stop_at_nan=True,
    ):
        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if (nvars) % 2:
            raise ProblemError('setup requires nvars = 2^p')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__((nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'nvars',
            'dw',
            'eps',
            'newton_maxiter',
            'newton_tol',
            'interval',
            'radius',
            'stop_at_nan',
            localVars=locals(),
            readOnly=True,
        )

        # compute dx and get discretization matrix A
        self.dx = (self.interval[1] - self.interval[0]) / self.nvars
        self.xvalues = np.array([self.interval[0] + i * self.dx for i in range(self.nvars)])

        self.A = problem_helper.get_finite_difference_matrix(
            derivative=2,
            order=2,
            type='center',
            dx=self.dx,
            size=self.nvars,
            dim=1,
            bc='periodic',
        )

        self.newton_itercount = 0
        self.lin_itercount = 0
        self.newton_ncalls = 0
        self.lin_ncalls = 0

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
            Current time (required here for the BC).

        Returns
        -------
        u : dtype_u
            The solution as mesh.
        """

        u = self.dtype_u(u0)
        eps2 = self.eps**2
        dw = self.dw

        Id = sp.eye(self.nvars)

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # print(n)
            # form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = (
                u
                - rhs
                - factor * (self.A.dot(u) - 2.0 / eps2 * u * (1.0 - u) * (1.0 - 2.0 * u) - 6.0 * dw * u * (1.0 - u))
            )

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (
                self.A
                - 2.0
                / eps2
                * sp.diags((1.0 - u) * (1.0 - 2.0 * u) - u * ((1.0 - 2.0 * u) + 2.0 * (1.0 - u)), offsets=0)
                - 6.0 * dw * sp.diags((1.0 - u) - u, offsets=0)
            )

            # newton update: u1 = u0 - g/dg
            u -= spsolve(dg, g)
            # u -= gmres(dg, g, x0=z, tol=self.lin_tol)[0]
            # increase iteration count
            n += 1

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.newton_maxiter:
            self.logger.warning('Newton did not converge after %i iterations, error is %s' % (n, res))

        self.newton_ncalls += 1
        self.newton_itercount += n

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me

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
        f = self.dtype_f(self.init)
        f[:] = self.A.dot(u) - 2.0 / self.eps**2 * u * (1.0 - u) * (1.0 - 2 * u) - 6.0 * self.dw * u * (1.0 - u)
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
            The exact solution.
        """

        v = 3.0 * np.sqrt(2) * self.eps * self.dw
        me = self.dtype_u(self.init, val=0.0)
        me[:] = 0.5 * (1 + np.tanh((self.radius - abs(self.xvalues) - v * t) / (np.sqrt(2) * self.eps)))
        return me


class allencahn_periodic_semiimplicit(allencahn_periodic_fullyimplicit):
    r"""
    This class implements the one-dimensional Allen-Cahn equation with driving force and periodic boundary conditions

    .. math::
        \frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2} - \frac{2}{\varepsilon^2} u (1 - u) (1 - 2u)
            - 6 d_w u (1 - u)

    for :math:`u \in [0, 1]`. For discretization of the Laplacian, centered finite differences are used.
    The exact solution is

    .. math::
        u(x, t) = 0.5 \left(1 + \tanh\left(\frac{r - |x| - vt}{\sqrt{2}\varepsilon}\right)\right)

    with :math:`v = 3 \sqrt{2} \varepsilon d_w` and radius :math:`r` of the circles. For time-stepping, the problem is treated
    in *semi-implicit* way, i.e., the part containing the Laplacian is treated implicitly, and the rest of the right-hand
    side is only evaluated at each time.
    """

    dtype_f = imex_mesh

    def __init__(
        self,
        nvars=128,
        dw=-0.04,
        eps=0.04,
        newton_maxiter=100,
        newton_tol=1e-12,
        interval=(-0.5, 0.5),
        radius=0.25,
        stop_at_nan=True,
    ):
        super().__init__(nvars, dw, eps, newton_maxiter, newton_tol, interval, radius, stop_at_nan)
        self.A -= sp.eye(self.init) * 0.0 / self.eps**2

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Simple linear solver for :math:`(I-factor\cdot A)\vec{u}=\vec{rhs}`.

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

        me = self.dtype_u(u0)
        me[:] = spsolve(sp.eye(self.nvars, format='csc') - factor * self.A, rhs)
        return me

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
        f = self.dtype_f(self.init)
        f.impl[:] = self.A.dot(u)
        f.expl[:] = (
            -2.0 / self.eps**2 * u * (1.0 - u) * (1.0 - 2.0 * u)
            - 6.0 * self.dw * u * (1.0 - u)
            + 0.0 / self.eps**2 * u
        )
        return f


class allencahn_periodic_multiimplicit(allencahn_periodic_fullyimplicit):
    r"""
    This class implements the one-dimensional Allen-Cahn equation with driving force and periodic boundary conditions

    .. math::
        \frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2} - \frac{2}{\varepsilon^2} u (1 - u) (1 - 2u)
            - 6 d_w u (1 - u)

    for :math:`u \in [0, 1]`. For discretization of the second order spatial derivative, centered finite differences are used.
    The exact solution is then given by

    .. math::
        u(x, t) = 0.5 \left(1 + \tanh\left(\frac{r - |x| - vt}{\sqrt{2}\varepsilon}\right)\right)

    with :math:`v = 3 \sqrt{2} \varepsilon d_w` and radius :math:`r` of the circles. For time-stepping, the problem is treated
    in a *multi-implicit* fashion, i.e., the nonlinear system containing the part with the Laplacian is solved with a
    linear solver provided by a ``SciPy`` routine, and the nonlinear system including the rest of the right-hand side is solved by
    Newton's method.
    """

    dtype_f = comp2_mesh

    def __init__(
        self,
        nvars=128,
        dw=-0.04,
        eps=0.04,
        newton_maxiter=100,
        newton_tol=1e-12,
        interval=(-0.5, 0.5),
        radius=0.25,
        stop_at_nan=True,
    ):
        super().__init__(nvars, dw, eps, newton_maxiter, newton_tol, interval, radius, stop_at_nan)
        self.A -= sp.eye(self.init) * 0.0 / self.eps**2

    def solve_system_1(self, rhs, factor, u0, t):
        r"""
        Simple linear solver for :math:`(I-factor\cdot A)\vec{u}=\vec{rhs}`.

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

        me = self.dtype_u(u0)
        me[:] = spsolve(sp.eye(self.nvars, format='csc') - factor * self.A, rhs)
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
            The right-hand side of the problem.
        """
        f = self.dtype_f(self.init)
        f.comp1[:] = self.A.dot(u)
        f.comp2[:] = (
            -2.0 / self.eps**2 * u * (1.0 - u) * (1.0 - 2.0 * u)
            - 6.0 * self.dw * u * (1.0 - u)
            + 0.0 / self.eps**2 * u
        )
        return f

    def solve_system_2(self, rhs, factor, u0, t):
        r"""
        Simple linear solver for :math:`(I-factor\cdot A)\vec{u}=\vec{rhs}`.

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

        u = self.dtype_u(u0)
        eps2 = self.eps**2
        dw = self.dw

        Id = sp.eye(self.nvars)

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # print(n)
            # form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = (
                u
                - rhs
                - factor
                * (-2.0 / eps2 * u * (1.0 - u) * (1.0 - 2.0 * u) - 6.0 * dw * u * (1.0 - u) + 0.0 / self.eps**2 * u)
            )

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (
                -2.0 / eps2 * sp.diags((1.0 - u) * (1.0 - 2.0 * u) - u * ((1.0 - 2.0 * u) + 2.0 * (1.0 - u)), offsets=0)
                - 6.0 * dw * sp.diags((1.0 - u) - u, offsets=0)
                + 0.0 / self.eps**2 * Id
            )

            # newton update: u1 = u0 - g/dg
            u -= spsolve(dg, g)
            # u -= gmres(dg, g, x0=z, tol=self.lin_tol)[0]
            # increase iteration count
            n += 1

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.newton_maxiter:
            self.logger.warning('Newton did not converge after %i iterations, error is %s' % (n, res))

        self.newton_ncalls += 1
        self.newton_itercount += n

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me
