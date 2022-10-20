import numpy as np

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh


# noinspection PyUnusedLocal
class logistics_equation(ptype):
    """
    Example implementing the logistic equation, taken from
    <https://www-users.cse.umn.edu/~olver/ln\_/odq.pdf> (Example 2.2)
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
        essential_keys = ['u0', 'lam', 'newton_maxiter', 'newton_tol', 'direct']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)
        problem_params['nvars'] = 1

        if 'stop_at_nan' not in problem_params:
            problem_params['stop_at_nan'] = True

        # invoke super init, passing dtype_u and dtype_f, plus setting number of elements to 2
        super(logistics_equation, self).__init__(
            (problem_params['nvars'], None, np.dtype('float64')), dtype_u, dtype_f, problem_params
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
        me[:] = (
            self.params.u0
            * np.exp(self.params.lam * t)
            / (1 - self.params.u0 + self.params.u0 * np.exp(self.params.lam * t))
        )
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
        f[:] = self.params.lam * u * (1 - u)
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

        if self.params.direct:

            d = (1 - dt * self.params.lam) ** 2 + 4 * dt * self.params.lam * rhs
            u = (-(1 - dt * self.params.lam) + np.sqrt(d)) / (2 * dt * self.params.lam)
            return u

        else:

            # start newton iteration
            n = 0
            res = 99
            while n < self.params.newton_maxiter:

                # form the function g with g(u) = 0
                g = u - dt * self.params.lam * u * (1 - u) - rhs

                # if g is close to 0, then we are done
                res = np.linalg.norm(g, np.inf)
                if res < self.params.newton_tol or np.isnan(res):
                    break

                # assemble dg/du
                dg = 1 - dt * self.params.lam * (1 - 2 * u)
                # newton update: u1 = u0 - g/dg
                u -= 1.0 / dg * g

                # increase iteration count
                n += 1

            if np.isnan(res) and self.params.stop_at_nan:
                raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
            elif np.isnan(res):
                self.logger.warning('Newton got nan after %i iterations...' % n)

            if n == self.params.newton_maxiter:
                raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

            return u
