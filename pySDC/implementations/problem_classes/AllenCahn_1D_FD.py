from __future__ import division

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, spsolve

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError, ProblemError

# http://www.personal.psu.edu/qud2/Res/Pre/dz09sisc.pdf


# noinspection PyUnusedLocal
class allencahn_fullyimplicit(ptype):
    """
    Example implementing the Allen-Cahn equation in 1D with finite differences and periodic BC

    Attributes:
        A: second-order FD discretization of the 1D laplace operator
        dx: distance between two spatial nodes
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
        essential_keys = ['nvars', 'nu', 'eps', 'newton_maxiter', 'newton_tol', 'radius']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if problem_params['nvars'] % 2 != 0:
            raise ProblemError('the setup requires nvars = 2^p per dimension')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(allencahn_fullyimplicit, self).__init__(problem_params['nvars'], dtype_u, dtype_f, problem_params)

        # compute dx and get discretization matrix A
        self.dx = 1.0 / self.params.nvars
        self.A = self.__get_A(self.params.nvars, self.params.nu, self.dx)
        self.xvalues = np.array([i * self.dx - 0.5 for i in range(self.params.nvars)])

        self.inner_solve_counter = 0
        self.newton_ncalls = 0

    @staticmethod
    def __get_A(N, nu, dx):
        """
        Helper function to assemble FD matrix A in sparse format

        Args:
            N (int): number of dofs
            nu (float): diffusion coefficient
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
        A *= nu / (dx ** 2)

        return A

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

        me = self.dtype_u(self.init)

        me.values[:] = u0.values
        nu = self.params.nu
        eps2 = self.params.eps ** 2

        Id = sp.eye(self.params.nvars)

        # start newton iteration
        n = 0
        res = 99
        while n < self.params.newton_maxiter:

            # form the function g with g(u) = 0
            g = me.values - factor * (self.A.dot(me.values) + 1.0 / eps2 * me.values * (1.0 - me.values ** nu))\
                - rhs.values

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.params.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (self.A + 1.0 / eps2 * sp.diags((1.0 - (nu + 1) * me.values ** nu), offsets=0))

            # newton update: u1 = u0 - g/dg
            me.values -= spsolve(dg, g)
            # increase iteration count
            n += 1
            # print(n, res)

        # if n == self.params.newton_maxiter:
        #     raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

        self.newton_ncalls += 1
        self.inner_solve_counter += n

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
        f.values = self.A.dot(u.values) + 1.0 / self.params.eps ** 2 * u.values * (1.0 - u.values ** self.params.nu)

        return f

    def eval_jacobian(self, u):
        """
        Evaluation of the Jacobian of the right-hand side

        Args:
            u: space values

        Returns:
            Jacobian matrix
        """

        dfdu = self.A + sp.diags(1.0 / self.params.eps ** 2 * (1.0 - (self.params.nu + 1) * u.values ** self.params.nu),
                                 offsets=0)
        return dfdu

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
        for i in range(self.params.nvars):
            me.values[i] = np.tanh((self.params.radius - abs(self.xvalues[i])) / (np.sqrt(2) * self.params.eps))

        return me
