import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype
from pySDC.helpers import problem_helper
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh, comp2_mesh


class allencahn_front_fullyimplicit(ptype):
    """
    Example implementing the Allen-Cahn equation in 1D with finite differences and inhomogeneous Dirichlet-BC,
    with driving force, 0-1 formulation (Bayreuth example)

    Attributes:
        A: second-order FD discretization of the 1D laplace operator
        dx: distance between two spatial nodes
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars, dw, eps, newton_maxiter, newton_tol, interval, stop_at_nan=True):

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if (nvars + 1) % 2 != 0:
            raise ProblemError('setup requires nvars = 2^p - 1')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__((nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'nvars', 'dw', 'eps', 'newton_maxiter', 'newton_tol', 'interval', 'stop_at_nan',
            localVars=locals(), readOnly=True)

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
            # form the function g with g(u) = 0
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
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
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
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        v = 3.0 * np.sqrt(2) * self.eps * self.dw
        me = self.dtype_u(self.init, val=0.0)
        me[:] = 0.5 * (1 + np.tanh((self.xvalues - v * t) / (np.sqrt(2) * self.eps)))
        return me


class allencahn_front_semiimplicit(allencahn_front_fullyimplicit):
    """
    Example implementing the Allen-Cahn equation in 1D with finite differences and inhomogeneous Dirichlet-BC,
    with driving force, 0-1 formulation (Bayreuth example), semi-implicit time-stepping

    Attributes:
        A: second-order FD discretization of the 1D laplace operator
        dx: distance between two spatial nodes
    """

    dtype_f = imex_mesh

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
        v = 3.0 * np.sqrt(2) * self.eps * self.dw
        self.uext[0] = 0.5 * (1 + np.tanh((self.interval[0] - v * t) / (np.sqrt(2) * self.eps)))
        self.uext[-1] = 0.5 * (1 + np.tanh((self.interval[1] - v * t) / (np.sqrt(2) * self.eps)))

        self.uext[1:-1] = u[:]

        f = self.dtype_f(self.init)
        f.impl[:] = self.A.dot(self.uext)[1:-1]
        f.expl[:] = -2.0 / self.eps**2 * u * (1.0 - u) * (1.0 - 2 * u) - 6.0 * self.dw * u * (1.0 - u)
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-factor*A)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        me = self.dtype_u(self.init)
        self.uext[0] = 0.0
        self.uext[-1] = 0.0
        self.uext[1:-1] = rhs[:]
        me[:] = spsolve(sp.eye(self.nvars + 2, format='csc') - factor * self.A, self.uext)[1:-1]
        return me


class allencahn_front_finel(allencahn_front_fullyimplicit):
    """
    Example implementing the Allen-Cahn equation in 1D with finite differences and inhomogeneous Dirichlet-BC,
    with driving force, 0-1 formulation (Bayreuth example), Finel's trick/parametrization
    """

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
            # form the function g with g(u) = 0
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
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
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
    """
    Example implementing the Allen-Cahn equation in 1D with finite differences and periodic BC,
    with driving force, 0-1 formulation (Bayreuth example)

    Attributes:
        A: second-order FD discretization of the 1D laplace operator
        dx: distance between two spatial nodes
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars, dw, eps, newton_maxiter, newton_tol, interval, radius, stop_at_nan=True):

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if (nvars) % 2 != 0:
            raise ProblemError('setup requires nvars = 2^p')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__((nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister('nvars', 'dw', 'eps', 'newton_maxiter', 'newton_tol', 'interval', 'radius',
                                       'stop_at_nan', localVars=locals(), readOnly=True)

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
        eps2 = self.eps**2
        dw = self.dw

        Id = sp.eye(self.nvars)

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # print(n)
            # form the function g with g(u) = 0
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
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """
        f = self.dtype_f(self.init)
        f[:] = (
            self.A.dot(u)
            - 2.0 / self.eps**2 * u * (1.0 - u) * (1.0 - 2 * u)
            - 6.0 * self.dw * u * (1.0 - u)
        )
        return f

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        v = 3.0 * np.sqrt(2) * self.eps * self.dw
        me = self.dtype_u(self.init, val=0.0)
        me[:] = 0.5 * (1 + np.tanh((self.radius - abs(self.xvalues) - v * t) / (np.sqrt(2) * self.eps)))
        return me


class allencahn_periodic_semiimplicit(allencahn_periodic_fullyimplicit):
    """
    Example implementing the Allen-Cahn equation in 1D with finite differences and periodic BC,
    with driving force, 0-1 formulation (Bayreuth example)
    """

    dtype_f = imex_mesh

    def __init__(self, nvars, dw, eps, newton_maxiter, newton_tol, interval, radius, stop_at_nan=True):
        super().__init__(nvars, dw, eps, newton_maxiter, newton_tol, interval, radius, stop_at_nan)
        self.A -= sp.eye(self.init) * 0.0 / self.eps**2

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-factor*A)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        me = self.dtype_u(u0)
        me[:] = spsolve(sp.eye(self.nvars, format='csc') - factor * self.A, rhs)
        return me

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
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
    """
    Example implementing the Allen-Cahn equation in 1D with finite differences and periodic BC,
    with driving force, 0-1 formulation (Bayreuth example)
    """

    dtype_f = comp2_mesh

    def __init__(self, nvars, dw, eps, newton_maxiter, newton_tol, interval, radius, stop_at_nan=True):
        super().__init__(nvars, dw, eps, newton_maxiter, newton_tol, interval, radius, stop_at_nan)
        self.A -= sp.eye(self.init) * 0.0 / self.eps**2

    def solve_system_1(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-factor*A)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        me = self.dtype_u(u0)
        me[:] = spsolve(sp.eye(self.nvars, format='csc') - factor * self.A, rhs)
        return me

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
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
        """
        Simple linear solver for (I-factor*A)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
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
            # form the function g with g(u) = 0
            g = (
                u
                - rhs
                - factor
                * (
                    -2.0 / eps2 * u * (1.0 - u) * (1.0 - 2.0 * u)
                    - 6.0 * dw * u * (1.0 - u)
                    + 0.0 / self.eps**2 * u
                )
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
