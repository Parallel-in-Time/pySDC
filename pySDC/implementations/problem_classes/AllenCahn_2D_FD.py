from __future__ import division

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, spsolve

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError, ProblemError

# http://www.personal.psu.edu/qud2/Res/Pre/dz09sisc.pdf


class allencahn_fullyimplicit(ptype):
    """
    Example implementing the Allen-Cahn equation in 2D with finite differences and periodic BC

    Attributes:
        A: second-order FD discretization of the 2D laplace operator
        dx: distance between two spatial nodes (same for both directions)
        xvalues: array of grid values
        newton_itercount (int): counts the number of Newton solves
        lin_itercount = counts the number of inner linear solves
        newton_ncalls = counts the number of Newton calls
        lin_ncalls = counts the number of inner linear calls
    """

    def __init__(self, problem_params, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'nu', 'eps', 'newton_maxiter', 'newton_tol', 'lin_tol', 'lin_maxiter', 'radius']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if len(problem_params['nvars']) != 2:
            raise ProblemError('this is a 2d example, got %s' % problem_params['nvars'])
        if problem_params['nvars'][0] != problem_params['nvars'][1]:
            raise ProblemError('need a square domain, got %s' % problem_params['nvars'])
        if problem_params['nvars'][0] % 2 != 0:
            raise ProblemError('the setup requires nvars = 2^p per dimension')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(allencahn_fullyimplicit, self).__init__(problem_params['nvars'], dtype_u, dtype_f, problem_params)

        # compute dx and get discretization matrix A
        self.dx = 1.0 / self.params.nvars[0]
        self.A = self.__get_A(self.params.nvars, self.dx)
        self.xvalues = np.array([i * self.dx - 0.5 for i in range(self.params.nvars[0])])

        self.newton_itercount = 0
        self.lin_itercount = 0
        self.newton_ncalls = 0
        self.lin_ncalls = 0

    @staticmethod
    def __get_A(N, dx):
        """
        Helper function to assemble FD matrix A in sparse format

        Args:
            N (list): number of dofs
            dx (float): distance between two spatial nodes

        Returns:
            scipy.sparse.csc_matrix: matrix A in CSC format
        """

        stencil = [1, -2, 1]
        zero_pos = 2
        dstencil = np.concatenate((stencil, np.delete(stencil, zero_pos - 1)))
        offsets = np.concatenate(([N[0] - i - 1 for i in reversed(range(zero_pos - 1))],
                                  [i - zero_pos + 1 for i in range(zero_pos - 1, len(stencil))]))
        doffsets = np.concatenate((offsets, np.delete(offsets, zero_pos - 1) - N[0]))

        A = sp.diags(dstencil, doffsets, shape=(N[0], N[0]), format='csc')
        A = sp.kron(A, sp.eye(N[0])) + sp.kron(sp.eye(N[1]), A)
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

        u = self.dtype_u(u0).values.flatten()
        z = self.dtype_u(self.init, val=0.0).values.flatten()
        nu = self.params.nu
        eps2 = self.params.eps ** 2

        Id = sp.eye(self.params.nvars[0] * self.params.nvars[1])

        # start newton iteration
        n = 0
        res = 99
        while n < self.params.newton_maxiter:

            # form the function g with g(u) = 0
            g = u - factor * (self.A.dot(u) + 1.0 / eps2 * u * (1.0 - u ** nu)) - rhs.values.flatten()

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.params.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (self.A + 1.0 / eps2 * sp.diags((1.0 - (nu + 1) * u ** nu), offsets=0))

            # newton update: u1 = u0 - g/dg
            # u -= spsolve(dg, g)
            u -= cg(dg, g, x0=z, tol=self.params.lin_tol, maxiter=self.params.lin_maxiter)[0]
            # increase iteration count
            n += 1

        # if n == self.params.newton_maxiter:
        #     raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

        me = self.dtype_u(self.init)
        me.values = u.reshape(self.params.nvars)

        self.newton_ncalls += 1
        self.newton_itercount += n

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
        v = u.values.flatten()
        f.values = self.A.dot(v) + 1.0 / self.params.eps ** 2 * v * (1.0 - v ** self.params.nu)
        f.values = f.values.reshape(self.params.nvars)

        return f

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        assert t == 0, 'ERROR: u_exact only valid for t=0'
        me = self.dtype_u(self.init, val=0.0)
        for i in range(self.params.nvars[0]):
            for j in range(self.params.nvars[1]):
                r2 = self.xvalues[i] ** 2 + self.xvalues[j] ** 2
                me.values[i, j] = np.tanh((self.params.radius - np.sqrt(r2)) / (np.sqrt(2) * self.params.eps))

        return me


# noinspection PyUnusedLocal
class allencahn_semiimplicit(allencahn_fullyimplicit):
    """
    Example implementing the Allen-Cahn equation in 2D with finite differences, SDC standard splitting

    Attributes:
        A: second-order FD discretization of the 2D laplace operator
        dx: distance between two spatial nodes (same for both directions)
    """

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
        v = u.values.flatten()
        f.impl.values = self.A.dot(v)
        f.expl.values = 1.0 / self.params.eps ** 2 * v * (1.0 - v ** self.params.nu)
        f.impl.values = f.impl.values.reshape(self.params.nvars)
        f.expl.values = f.expl.values.reshape(self.params.nvars)

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

        class context:
            num_iter = 0

        def callback(xk):
            context.num_iter += 1
            return context.num_iter

        me = self.dtype_u(self.init)

        Id = sp.eye(self.params.nvars[0] * self.params.nvars[1])

        me.values = cg(Id - factor * self.A, rhs.values.flatten(), x0=u0.values.flatten(), tol=self.params.lin_tol,
                       maxiter=self.params.lin_maxiter, callback=callback)[0]
        me.values = me.values.reshape(self.params.nvars)

        self.lin_ncalls += 1
        self.lin_itercount += context.num_iter

        return me


# noinspection PyUnusedLocal
class allencahn_semiimplicit_v2(allencahn_fullyimplicit):
    """
    Example implementing the Allen-Cahn equation in 2D with finite differences, AC splitting

    Attributes:
        A: second-order FD discretization of the 2D laplace operator
        dx: distance between two spatial nodes (same for both directions)
    """

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
        v = u.values.flatten()
        f.impl.values = self.A.dot(v) - 1.0 / self.params.eps ** 2 * v ** (self.params.nu + 1)
        f.expl.values = 1.0 / self.params.eps ** 2 * v
        f.impl.values = f.impl.values.reshape(self.params.nvars)
        f.expl.values = f.expl.values.reshape(self.params.nvars)

        return f

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

        u = self.dtype_u(u0).values.flatten()
        z = self.dtype_u(self.init, val=0.0).values.flatten()
        nu = self.params.nu
        eps2 = self.params.eps ** 2

        Id = sp.eye(self.params.nvars[0] * self.params.nvars[1])

        # start newton iteration
        n = 0
        res = 99
        while n < self.params.newton_maxiter:

            # form the function g with g(u) = 0
            g = u - factor * (self.A.dot(u) - 1.0 / eps2 * u ** (nu + 1)) - rhs.values.flatten()

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.params.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (self.A - 1.0 / eps2 * sp.diags(((nu + 1) * u ** nu), offsets=0))

            # newton update: u1 = u0 - g/dg
            # u -= spsolve(dg, g)
            u -= cg(dg, g, x0=z, tol=self.params.lin_tol)[0]
            # increase iteration count
            n += 1
            # print(n, res)

        # if n == self.params.newton_maxiter:
        #     raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

        me = self.dtype_u(self.init)
        me.values = u.reshape(self.params.nvars)

        self.newton_ncalls += 1
        self.newton_itercount += n

        return me


# noinspection PyUnusedLocal
class allencahn_multiimplicit(allencahn_fullyimplicit):
    """
    Example implementing the Allen-Cahn equation in 2D with finite differences, SDC standard splitting

    Attributes:
        A: second-order FD discretization of the 2D laplace operator
        dx: distance between two spatial nodes (same for both directions)
    """

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
        v = u.values.flatten()
        f.comp1.values = self.A.dot(v)
        f.comp2.values = 1.0 / self.params.eps ** 2 * v * (1.0 - v ** self.params.nu)
        f.comp1.values = f.comp1.values.reshape(self.params.nvars)
        f.comp2.values = f.comp2.values.reshape(self.params.nvars)

        return f

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

        class context:
            num_iter = 0

        def callback(xk):
            context.num_iter += 1
            return context.num_iter

        me = self.dtype_u(self.init)

        Id = sp.eye(self.params.nvars[0] * self.params.nvars[1])

        me.values = cg(Id - factor * self.A, rhs.values.flatten(), x0=u0.values.flatten(), tol=self.params.lin_tol,
                       maxiter=self.params.lin_maxiter, callback=callback)[0]
        me.values = me.values.reshape(self.params.nvars)

        self.lin_ncalls += 1
        self.lin_itercount += context.num_iter

        return me

    def solve_system_2(self, rhs, factor, u0, t):
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

        u = self.dtype_u(u0).values.flatten()
        z = self.dtype_u(self.init, val=0.0).values.flatten()
        nu = self.params.nu
        eps2 = self.params.eps ** 2

        Id = sp.eye(self.params.nvars[0] * self.params.nvars[1])

        # start newton iteration
        n = 0
        res = 99
        while n < self.params.newton_maxiter:

            # form the function g with g(u) = 0
            g = u - factor * (1.0 / eps2 * u * (1.0 - u ** nu)) - rhs.values.flatten()

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.params.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (1.0 / eps2 * sp.diags((1.0 - (nu + 1) * u ** nu), offsets=0))

            # newton update: u1 = u0 - g/dg
            # u -= spsolve(dg, g)
            u -= cg(dg, g, x0=z, tol=self.params.lin_tol)[0]
            # increase iteration count
            n += 1
            # print(n, res)

        # if n == self.params.newton_maxiter:
        #     raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

        me = self.dtype_u(self.init)
        me.values = u.reshape(self.params.nvars)

        self.newton_ncalls += 1
        self.newton_itercount += n

        return me


# noinspection PyUnusedLocal
class allencahn_multiimplicit_v2(allencahn_fullyimplicit):
    """
    Example implementing the Allen-Cahn equation in 2D with finite differences, AC splitting

    Attributes:
        A: second-order FD discretization of the 2D laplace operator
        dx: distance between two spatial nodes (same for both directions)
    """

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
        v = u.values.flatten()
        f.comp1.values = self.A.dot(v) - 1.0 / self.params.eps ** 2 * v ** (self.params.nu + 1)
        f.comp2.values = 1.0 / self.params.eps ** 2 * v
        f.comp1.values = f.comp1.values.reshape(self.params.nvars)
        f.comp2.values = f.comp2.values.reshape(self.params.nvars)

        return f

    def solve_system_1(self, rhs, factor, u0, t):
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

        u = self.dtype_u(u0).values.flatten()
        z = self.dtype_u(self.init, val=0.0).values.flatten()
        nu = self.params.nu
        eps2 = self.params.eps ** 2

        Id = sp.eye(self.params.nvars[0] * self.params.nvars[1])

        # start newton iteration
        n = 0
        res = 99
        while n < self.params.newton_maxiter:

            # form the function g with g(u) = 0
            g = u - factor * (self.A.dot(u) - 1.0 / eps2 * u ** (nu + 1)) - rhs.values.flatten()

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.params.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (self.A - 1.0 / eps2 * sp.diags(((nu + 1) * u ** nu), offsets=0))

            # newton update: u1 = u0 - g/dg
            # u -= spsolve(dg, g)
            u -= cg(dg, g, x0=z, tol=self.params.lin_tol)[0]
            # increase iteration count
            n += 1
            # print(n, res)

        # if n == self.params.newton_maxiter:
        #     raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

        me = self.dtype_u(self.init)
        me.values = u.reshape(self.params.nvars)

        self.newton_ncalls += 1
        self.newton_itercount += n

        return me

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

        me = self.dtype_u(self.init)

        me.values = 1.0 / (1.0 - factor * 1.0 / self.params.eps ** 2) * rhs.values
        me.values = me.values.reshape(self.params.nvars)
        return me
