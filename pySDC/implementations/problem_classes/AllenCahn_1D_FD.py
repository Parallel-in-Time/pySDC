
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.parallel_mesh import parallel_mesh, parallel_imex_mesh, parallel_comp2_mesh


class allencahn_front_fullyimplicit(ptype):
    """
    Example implementing the Allen-Cahn equation in 1D with finite differences and inhomogeneous Dirichlet-BC,
    with driving force, 0-1 formulation (Bayreuth example)

    Attributes:
        A: second-order FD discretization of the 1D laplace operator
        dx: distance between two spatial nodes
    """

    def __init__(self, problem_params, dtype_u=parallel_mesh, dtype_f=parallel_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'dw', 'eps', 'newton_maxiter', 'newton_tol', 'interval']
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
        super(allencahn_front_fullyimplicit, self).__init__((problem_params['nvars'], None, np.dtype('float64')),
                                                            dtype_u, dtype_f, problem_params)

        # compute dx and get discretization matrix A
        self.dx = (self.params.interval[1] - self.params.interval[0]) / (self.params.nvars + 1)
        self.xvalues = np.array([(i + 1 - (self.params.nvars + 1) / 2) * self.dx for i in range(self.params.nvars)])

        self.A = self.__get_A(self.params.nvars, self.dx)
        self.uext = self.dtype_u((self.init[0] + 2, self.init[1], self.init[2]), val=0.0)

        self.newton_itercount = 0
        self.lin_itercount = 0
        self.newton_ncalls = 0
        self.lin_ncalls = 0

    @staticmethod
    def __get_A(N, dx):
        """
        Helper function to assemble FD matrix A in sparse format

        Args:
            N (int): number of dofs
            dx (float): distance between two spatial nodes

        Returns:
            scipy.sparse.csc_matrix: matrix A in CSC format
        """

        stencil = [1, -2, 1]
        A = sp.diags(stencil, [-1, 0, 1], shape=(N + 2, N + 2), format='lil')
        A *= 1.0 / (dx ** 2)

        return A

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
        eps2 = self.params.eps ** 2
        dw = self.params.dw

        Id = sp.eye(self.params.nvars)

        v = 3.0 * np.sqrt(2) * self.params.eps * self.params.dw
        self.uext[0] = 0.5 * (1 + np.tanh((self.params.interval[0] - v * t) / (np.sqrt(2) * self.params.eps)))
        self.uext[-1] = 0.5 * (1 + np.tanh((self.params.interval[1] - v * t) / (np.sqrt(2) * self.params.eps)))

        A = self.A[1:-1, 1:-1]
        # start newton iteration
        n = 0
        res = 99
        while n < self.params.newton_maxiter:
            # print(n)
            # form the function g with g(u) = 0
            self.uext[1:-1] = u[:]
            g = u - rhs \
                - factor * (self.A.dot(self.uext)[1:-1] - 2.0 / eps2 * u * (1.0 - u) * (1.0 - 2.0 * u) -
                            6.0 * dw * u * (1.0 - u))

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.params.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (A - 2.0 / eps2 * sp.diags(
                (1.0 - u) * (1.0 - 2.0 * u) - u * ((1.0 - 2.0 * u) + 2.0 * (1.0 - u)), offsets=0) - 6.0 * dw * sp.diags(
                (1.0 - u) - u, offsets=0))

            # newton update: u1 = u0 - g/dg
            u -= spsolve(dg, g)
            # u -= gmres(dg, g, x0=z, tol=self.params.lin_tol)[0]
            # increase iteration count
            n += 1

        if np.isnan(res) and self.params.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.params.newton_maxiter:
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
        v = 3.0 * np.sqrt(2) * self.params.eps * self.params.dw
        self.uext[0] = 0.5 * (1 + np.tanh((self.params.interval[0] - v * t) / (np.sqrt(2) * self.params.eps)))
        self.uext[-1] = 0.5 * (1 + np.tanh((self.params.interval[1] - v * t) / (np.sqrt(2) * self.params.eps)))

        self.uext[1:-1] = u[:]

        f = self.dtype_f(self.init)
        f[:] = self.A.dot(self.uext)[1:-1] - \
            2.0 / self.params.eps ** 2 * u * (1.0 - u) * (1.0 - 2 * u) - \
            6.0 * self.params.dw * u * (1.0 - u)
        return f

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        v = 3.0 * np.sqrt(2) * self.params.eps * self.params.dw
        me = self.dtype_u(self.init, val=0.0)
        me[:] = 0.5 * (1 + np.tanh((self.xvalues - v * t) / (np.sqrt(2) * self.params.eps)))
        return me


class allencahn_front_semiimplicit(allencahn_front_fullyimplicit):
    """
    Example implementing the Allen-Cahn equation in 1D with finite differences and inhomogeneous Dirichlet-BC,
    with driving force, 0-1 formulation (Bayreuth example), semi-implicit time-stepping

    Attributes:
        A: second-order FD discretization of the 1D laplace operator
        dx: distance between two spatial nodes
    """

    def __init__(self, problem_params, dtype_u=parallel_mesh, dtype_f=parallel_imex_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type with implicit and explicit components (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'dw', 'eps', 'newton_maxiter', 'newton_tol', 'interval']
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
        super(allencahn_front_semiimplicit, self).__init__(problem_params, dtype_u, dtype_f)

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
        v = 3.0 * np.sqrt(2) * self.params.eps * self.params.dw
        self.uext[0] = 0.5 * (1 + np.tanh((self.params.interval[0] - v * t) / (np.sqrt(2) * self.params.eps)))
        self.uext[-1] = 0.5 * (1 + np.tanh((self.params.interval[1] - v * t) / (np.sqrt(2) * self.params.eps)))

        self.uext[1:-1] = u[:]

        f = self.dtype_f(self.init)
        f.impl[:] = self.A.dot(self.uext)[1:-1]
        f.expl[:] = - 2.0 / self.params.eps ** 2 * u * (1.0 - u) * (1.0 - 2 * u) - \
            6.0 * self.params.dw * u * (1.0 - u)
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
        me[:] = spsolve(sp.eye(self.params.nvars + 2, format='csc') - factor * self.A, self.uext)[1:-1]
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
        dw = self.params.dw
        a2 = np.tanh(self.dx / (np.sqrt(2) * self.params.eps)) ** 2

        Id = sp.eye(self.params.nvars)

        v = 3.0 * np.sqrt(2) * self.params.eps * self.params.dw
        self.uext[0] = 0.5 * (1 + np.tanh((self.params.interval[0] - v * t) / (np.sqrt(2) * self.params.eps)))
        self.uext[-1] = 0.5 * (1 + np.tanh((self.params.interval[1] - v * t) / (np.sqrt(2) * self.params.eps)))

        A = self.A[1:-1, 1:-1]
        # start newton iteration
        n = 0
        res = 99
        while n < self.params.newton_maxiter:
            # print(n)
            # form the function g with g(u) = 0
            self.uext[1:-1] = u[:]
            gprim = 1.0 / self.dx ** 2 * ((1.0 - a2) / (1.0 - a2 * (2.0 * u - 1.0) ** 2) - 1.0) * (2.0 * u - 1.0)
            g = u - rhs - factor * (self.A.dot(self.uext)[1:-1] - 1.0 * gprim - 6.0 * dw * u * (1.0 - u))

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.params.newton_tol:
                break

            # assemble dg
            dgprim = 1.0 / self.dx ** 2 * \
                (2.0 * ((1.0 - a2) / (1.0 - a2 * (2.0 * u - 1.0) ** 2) - 1.0) +
                 (2.0 * u - 1) ** 2 * (1.0 - a2) * 4 * a2 / (1.0 - a2 * (2.0 * u - 1.0) ** 2) ** 2)

            dg = Id - factor * (A - 1.0 * sp.diags(dgprim, offsets=0) - 6.0 * dw * sp.diags((1.0 - u) - u, offsets=0))

            # newton update: u1 = u0 - g/dg
            u -= spsolve(dg, g)
            # For some reason, doing cg or gmres does not work so well here...
            # u -= cg(dg, g, x0=z, tol=self.params.lin_tol)[0]
            # increase iteration count
            n += 1

        if np.isnan(res) and self.params.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.params.newton_maxiter:
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
        v = 3.0 * np.sqrt(2) * self.params.eps * self.params.dw
        self.uext[0] = 0.5 * (1 + np.tanh((self.params.interval[0] - v * t) / (np.sqrt(2) * self.params.eps)))
        self.uext[-1] = 0.5 * (1 + np.tanh((self.params.interval[1] - v * t) / (np.sqrt(2) * self.params.eps)))

        self.uext[1:-1] = u[:]

        a2 = np.tanh(self.dx / (np.sqrt(2) * self.params.eps)) ** 2
        gprim = 1.0 / self.dx ** 2 * ((1.0 - a2) / (1.0 - a2 * (2.0 * u - 1.0) ** 2) - 1) \
            * (2.0 * u - 1.0)
        f = self.dtype_f(self.init)
        f[:] = self.A.dot(self.uext)[1:-1] - 1.0 * gprim - 6.0 * self.params.dw * u * (1.0 - u)
        return f


class allencahn_periodic_fullyimplicit(ptype):
    """
    Example implementing the Allen-Cahn equation in 1D with finite differences and periodic BC,
    with driving force, 0-1 formulation (Bayreuth example)

    Attributes:
        A: second-order FD discretization of the 1D laplace operator
        dx: distance between two spatial nodes
    """

    def __init__(self, problem_params, dtype_u=parallel_mesh, dtype_f=parallel_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'dw', 'eps', 'newton_maxiter', 'newton_tol', 'interval', 'radius']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if (problem_params['nvars']) % 2 != 0:
            raise ProblemError('setup requires nvars = 2^p')

        if 'stop_at_nan' not in problem_params:
            problem_params['stop_at_nan'] = True

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(allencahn_periodic_fullyimplicit, self).__init__((problem_params['nvars'], None, np.dtype('float64')),
                                                               dtype_u, dtype_f, problem_params)

        # compute dx and get discretization matrix A
        self.dx = (self.params.interval[1] - self.params.interval[0]) / self.params.nvars
        self.xvalues = np.array([self.params.interval[0] + i * self.dx for i in range(self.params.nvars)])

        self.A = self.__get_A(self.params.nvars, self.dx)

        self.newton_itercount = 0
        self.lin_itercount = 0
        self.newton_ncalls = 0
        self.lin_ncalls = 0

    @staticmethod
    def __get_A(N, dx):
        """
        Helper function to assemble FD matrix A in sparse format

        Args:
            N (int): number of dofs
            dx (float): distance between two spatial nodes

        Returns:
            scipy.sparse.csc_matrix: matrix A in CSC format
        """

        stencil = [1, -2, 1]
        zero_pos = 2
        dstencil = np.concatenate((stencil, np.delete(stencil, zero_pos - 1)))
        offsets = np.concatenate(([N - i - 1 for i in reversed(range(zero_pos - 1))],
                                  [i - zero_pos + 1 for i in range(zero_pos - 1, len(stencil))]))
        doffsets = np.concatenate((offsets, np.delete(offsets, zero_pos - 1) - N))

        A = sp.diags(dstencil, doffsets, shape=(N, N), format='csc')
        A *= 1.0 / (dx ** 2)

        return A

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
        eps2 = self.params.eps ** 2
        dw = self.params.dw

        Id = sp.eye(self.params.nvars)

        # start newton iteration
        n = 0
        res = 99
        while n < self.params.newton_maxiter:
            # print(n)
            # form the function g with g(u) = 0
            g = u - rhs - factor * (self.A.dot(u) - 2.0 / eps2 * u * (1.0 - u) * (1.0 - 2.0 * u) -
                                    6.0 * dw * u * (1.0 - u))

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.params.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (self.A - 2.0 / eps2 * sp.diags(
                (1.0 - u) * (1.0 - 2.0 * u) - u * ((1.0 - 2.0 * u) + 2.0 * (1.0 - u)), offsets=0) - 6.0 * dw * sp.diags(
                (1.0 - u) - u, offsets=0))

            # newton update: u1 = u0 - g/dg
            u -= spsolve(dg, g)
            # u -= gmres(dg, g, x0=z, tol=self.params.lin_tol)[0]
            # increase iteration count
            n += 1

        if np.isnan(res) and self.params.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.params.newton_maxiter:
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
        f[:] = self.A.dot(u) - 2.0 / self.params.eps ** 2 * u * (1.0 - u) * (1.0 - 2 * u) - \
            6.0 * self.params.dw * u * (1.0 - u)
        return f

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        v = 3.0 * np.sqrt(2) * self.params.eps * self.params.dw
        me = self.dtype_u(self.init, val=0.0)
        me[:] = 0.5 * (1 + np.tanh((self.params.radius - abs(self.xvalues) - v * t) / (np.sqrt(2) * self.params.eps)))
        return me


class allencahn_periodic_semiimplicit(allencahn_periodic_fullyimplicit):
    """
    Example implementing the Allen-Cahn equation in 1D with finite differences and periodic BC,
    with driving force, 0-1 formulation (Bayreuth example)
    """

    def __init__(self, problem_params, dtype_u=parallel_mesh, dtype_f=parallel_imex_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'dw', 'eps', 'newton_maxiter', 'newton_tol', 'interval', 'radius']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if (problem_params['nvars']) % 2 != 0:
            raise ProblemError('setup requires nvars = 2^p')

        if 'stop_at_nan' not in problem_params:
            problem_params['stop_at_nan'] = True

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(allencahn_periodic_semiimplicit, self).__init__(problem_params, dtype_u, dtype_f)

        self.A -= sp.eye(self.init) * 0.0 / self.params.eps ** 2

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
        me[:] = spsolve(sp.eye(self.params.nvars, format='csc') - factor * self.A, rhs)
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
        f.expl[:] = -2.0 / self.params.eps ** 2 * u * (1.0 - u) * (1.0 - 2.0 * u) - \
            6.0 * self.params.dw * u * (1.0 - u) + 0.0 / self.params.eps ** 2 * u
        return f


class allencahn_periodic_multiimplicit(allencahn_periodic_fullyimplicit):
    """
    Example implementing the Allen-Cahn equation in 1D with finite differences and periodic BC,
    with driving force, 0-1 formulation (Bayreuth example)
    """

    def __init__(self, problem_params, dtype_u=parallel_mesh, dtype_f=parallel_comp2_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'dw', 'eps', 'newton_maxiter', 'newton_tol', 'interval', 'radius']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if (problem_params['nvars']) % 2 != 0:
            raise ProblemError('setup requires nvars = 2^p')

        if 'stop_at_nan' not in problem_params:
            problem_params['stop_at_nan'] = True

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(allencahn_periodic_multiimplicit, self).__init__(problem_params, dtype_u, dtype_f)

        self.A -= sp.eye(self.init) * 0.0 / self.params.eps ** 2

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
        me[:] = spsolve(sp.eye(self.params.nvars, format='csc') - factor * self.A, rhs)
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
        f.comp2[:] = -2.0 / self.params.eps ** 2 * u * (1.0 - u) * (1.0 - 2.0 * u) - \
            6.0 * self.params.dw * u * (1.0 - u) + 0.0 / self.params.eps ** 2 * u
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
        eps2 = self.params.eps ** 2
        dw = self.params.dw

        Id = sp.eye(self.params.nvars)

        # start newton iteration
        n = 0
        res = 99
        while n < self.params.newton_maxiter:
            # print(n)
            # form the function g with g(u) = 0
            g = u - rhs - factor * (- 2.0 / eps2 * u * (1.0 - u) * (1.0 - 2.0 * u) -
                                    6.0 * dw * u * (1.0 - u) + 0.0 / self.params.eps ** 2 * u)

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.params.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (- 2.0 / eps2 * sp.diags(
                (1.0 - u) * (1.0 - 2.0 * u) - u * ((1.0 - 2.0 * u) + 2.0 * (1.0 - u)), offsets=0) - 6.0 * dw * sp.diags(
                (1.0 - u) - u, offsets=0) + 0.0 / self.params.eps ** 2 * Id)

            # newton update: u1 = u0 - g/dg
            u -= spsolve(dg, g)
            # u -= gmres(dg, g, x0=z, tol=self.params.lin_tol)[0]
            # increase iteration count
            n += 1

        if np.isnan(res) and self.params.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.params.newton_maxiter:
            self.logger.warning('Newton did not converge after %i iterations, error is %s' % (n, res))

        self.newton_ncalls += 1
        self.newton_itercount += n

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me
