import numpy as np

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh


# noinspection PyUnusedLocal
class vanderpol(ptype):
    """
    Example implementing the van der pol oscillator
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
        essential_keys = ['u0', 'mu', 'newton_maxiter', 'newton_tol']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)
        problem_params['nvars'] = 2

        if 'stop_at_nan' not in problem_params:
            problem_params['stop_at_nan'] = True

        # invoke super init, passing dtype_u and dtype_f, plus setting number of elements to 2
        super(vanderpol, self).__init__((problem_params['nvars'], None, np.dtype('float64')),
                                        dtype_u, dtype_f, problem_params)

    def u_exact(self, t):
        """
        Dummy routine for the exact solution, currently only passes the initial values

        Args:
            t (float): current time
        Returns:
            dtype_u: mesh type containing the initial values
        """

        # thou shall not call this at time > 0

        me = self.dtype_u(self.init)
        me[:] = self.params.u0[:]
        return me

    def eval_f(self, u, t):
        """
        Routine to compute the RHS for both components simultaneously

        Args:
            u (dtype_u): the current values
            t (float): current time (not used here)
        Returns:
            dtype_f: RHS, 2 components
        """

        x1 = u[0]
        x2 = u[1]
        f = self.dtype_f(self.init)
        f[0] = x2
        f[1] = self.params.mu * (1 - x1 ** 2) * x2 - x1
        return f

    def solve_system(self, rhs, dt, u0, t):
        """
        Simple Newton solver for the nonlinear system

        Args:
            rhs (dtype_f): right-hand side for the nonlinear system
            dt (float): abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution u
        """

        mu = self.params.mu

        # create new mesh object from u0 and set initial values for iteration
        u = self.dtype_u(u0)
        x1 = u[0]
        x2 = u[1]

        # start newton iteration
        n = 0
        res = 99
        while n < self.params.newton_maxiter:

            # form the function g with g(u) = 0
            g = np.array([x1 - dt * x2 - rhs[0], x2 - dt * (mu * (1 - x1 ** 2) * x2 - x1) - rhs[1]])

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.params.newton_tol or np.isnan(res):
                break

            # prefactor for dg/du
            c = 1.0 / (-2 * dt ** 2 * mu * x1 * x2 - dt ** 2 - 1 + dt * mu * (1 - x1 ** 2))
            # assemble dg/du
            dg = c * np.array([[dt * mu * (1 - x1 ** 2) - 1, -dt], [2 * dt * mu * x1 * x2 + dt, -1]])

            # newton update: u1 = u0 - g/dg
            u -= np.dot(dg, g)

            # set new values and increase iteration count
            x1 = u[0]
            x2 = u[1]
            n += 1

        if np.isnan(res) and self.params.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.params.newton_maxiter:
            raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

        return u
