import numpy as np

from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh


# noinspection PyUnusedLocal
class vanderpol(ptype):
    r"""
    This class implements the stiff Van der Pol oscillator given by the equation

    .. math::
        \frac{d^2 u(t)}{d t^2} - \mu (1 - u(t)^2) \frac{d u(t)}{dt} + u(t) = 0.

    Parameters
    ----------
    u0 : sequence of array_like, optional
        Initial condition.
    mu : float, optional
        Stiff parameter :math:`\mu`.
    newton_maxiter : int, optional
        Maximum number of iterations for Newton's method to terminate.
    newton_tol : float, optional
        Tolerance for Newton to terminate.
    stop_at_nan : bool, optional
        Indicate whether Newton's method should stop if nan values arise.
    crash_at_maxiter = bool, optional
        Indicates whether Newton's method should stop if maximum number of iterations
        `newton_maxiter` is reached.

    Attributes
    ----------
    work_counters : WorkCounter
        Counts different things, here: Number of evaluations of the right-hand side in `eval_f`
        and number of Newton calls in each Newton iterations are counted.
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, u0=None, mu=5.0, newton_maxiter=100, newton_tol=1e-9, stop_at_nan=True, crash_at_maxiter=True):
        """Initialization routine"""
        nvars = 2

        if u0 is None:
            u0 = [2.0, 0.0]

        super().__init__((nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister('nvars', 'u0', localVars=locals(), readOnly=True)
        self._makeAttributeAndRegister(
            'mu', 'newton_maxiter', 'newton_tol', 'stop_at_nan', 'crash_at_maxiter', localVars=locals()
        )
        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

    def u_exact(self, t, u_init=None, t_init=None):
        r"""
        Routine to approximate the exact solution at time t by scipy or give initial conditions when called at :math:`t=0`.

        Parameters
        ----------
        t : float
            Current time.
        u_init : pySDC.problem.vanderpol.dtype_u
            Initial conditions for getting the exact solution.
        t_init : float
            The starting time.

        Returns
        -------
        me : dtype_u
            Approximate exact solution.
        """

        me = self.dtype_u(self.init)

        if t > 0.0:

            def eval_rhs(t, u):
                return self.eval_f(u, t)

            me[:] = self.generate_scipy_reference_solution(eval_rhs, t, u_init, t_init)
        else:
            me[:] = self.u0
        return me

    def eval_f(self, u, t):
        """
        Routine to compute the right-hand side for both components simultaneously.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time at which the numerical solution is computed (not used here).

        Returns
        -------
        f : dtype_f
            The right-hand side (contains 2 components).
        """

        x1 = u[0]
        x2 = u[1]
        f = self.f_init
        f[0] = x2
        f[1] = self.mu * (1 - x1**2) * x2 - x1
        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, dt, u0, t):
        """
        Simple Newton solver for the nonlinear system.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        dt : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        u : dtype_u
            The solution u.
        """

        mu = self.mu

        # create new mesh object from u0 and set initial values for iteration
        u = self.dtype_u(u0)
        x1 = u[0]
        x2 = u[1]

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # form the function g with g(u) = 0
            g = np.array([x1 - dt * x2 - rhs[0], x2 - dt * (mu * (1 - x1**2) * x2 - x1) - rhs[1]])

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol or np.isnan(res):
                break

            # prefactor for dg/du
            c = 1.0 / (-2 * dt**2 * mu * x1 * x2 - dt**2 - 1 + dt * mu * (1 - x1**2))
            # assemble dg/du
            dg = c * np.array([[dt * mu * (1 - x1**2) - 1, -dt], [2 * dt * mu * x1 * x2 + dt, -1]])

            # newton update: u1 = u0 - g/dg
            u -= np.dot(dg, g)

            # set new values and increase iteration count
            x1 = u[0]
            x2 = u[1]
            n += 1
            self.work_counters['newton']()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.newton_maxiter and self.crash_at_maxiter:
            raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

        return u
