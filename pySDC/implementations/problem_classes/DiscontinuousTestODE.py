import numpy as np

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh


class DiscontinuousTestODE(ptype):
    r"""
    This class implements a very simple test case of a ordinary differential equation consisting of one discrete event. The dynamics of
    the solution changes when the state function :math:`h(u) := u - 5` changes the sign. The problem is defined by:

    if :math:`u - 5 < 0:`

        .. math::
            \fra{d u}{dt} = u

    else:

        .. math::
            \frac{d u}{dt} = \frac{4}{t^*},

    where :math:`t^* = \log(5) \approx 1.6094379`. For :math:`h(u) < 0`, i.e., :math:`t \leq t^*` the exact solution is
    :math:`u(t) = exp(t)`; for :math:`h(u) \geq 0`, i.e., :math:`t \geq t^*` the exact solution is :math:`u(t) = \frac{4 t}{t^*} + 1`.

    Attributes
    ----------
    t_switch_exact : float
        Exact event time with :math:`t^* = \log(5)`.
    t_switch: float
        Time point of the discrete event found by switch estimation.
    nswitches: int
        Number of switches found by switch estimation.
    newton_itercount: int
        Counts the number of Newton iterations.
    newton_ncalls: int
        Counts the number of how often Newton is called in the simulation of the problem.
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, newton_maxiter=100, newton_tol=1e-8):
        """Initialization routine"""
        nvars = 1
        super().__init__(init=(nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister('nvars', localVars=locals(), readOnly=True)
        self._makeAttributeAndRegister('newton_maxiter', 'newton_tol', localVars=locals())

        if self.nvars != 1:
            raise ParameterError('nvars has to be equal to 1!')

        self.t_switch_exact = np.log(5)
        self.t_switch = None
        self.nswitches = 0
        self.newton_itercount = 0
        self.newton_ncalls = 0

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

        t_switch = np.inf if self.t_switch is None else self.t_switch

        f = self.dtype_f(self.init, val=0.0)
        h = u[0] - 5
        if h >= 0 or t >= t_switch:
            f[:] = 4 / self.t_switch_exact
        else:
            f[:] = u
        return f

    def solve_system(self, rhs, dt, u0, t):
        r"""
        Simple Newton solver for :math:`(I-factor\cdot A)\vec{u}=\vec{rhs}`.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        dt : float
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

        t_switch = np.inf if self.t_switch is None else self.t_switch

        h = rhs[0] - 5
        u = self.dtype_u(u0)

        n = 0
        res = 99
        while n < self.newton_maxiter:
            # form function g with g(u) = 0
            if h >= 0 or t >= t_switch:
                g = u - dt * (4 / self.t_switch_exact) - rhs
            else:
                g = u - dt * u - rhs

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.newton_tol:
                break

            if h >= 0 or t >= t_switch:
                dg = 1
            else:
                dg = 1 - dt

            # newton update
            u -= 1.0 / dg * g

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

    def u_exact(self, t, u_init=None, t_init=None):
        """
        Routine to compute the exact solution at time t.

        Parameters
        ----------
        t : float
            Time of the exact solution.
        u_init : pySDC.problem.DiscontinuousTestODE.dtype_u
            Initial conditions for getting the exact solution.
        t_init : float
            The starting time.

        Returns
        -------
        me : dtype_u
            The exact solution.
        """

        me = self.dtype_u(self.init)
        if t <= self.t_switch_exact:
            me[:] = np.exp(t)
        else:
            me[:] = (4 * t) / self.t_switch_exact + 1
        return me

    def get_switching_info(self, u, t):
        """
        Provides information about the state function of the problem. When the state function changes its sign,
        typically an event occurs. So the check for an event should be done in the way that the state function
        is checked for a sign change. If this is the case, the intermediate value theorem states a root in this
        step.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution at time t.
        t : float
            Current time of the numerical solution.

        Returns
        -------
        switch_detected : bool
            Indicates whether a discrete event is found or not.
        m_guess : int
            The index before the sign changes.
        state_function : list
            Defines the values of the state function at collocation nodes where it changes the sign.
        """

        switch_detected = False
        m_guess = -100

        for m in range(1, len(u)):
            h_prev_node = u[m - 1][0] - 5
            h_curr_node = u[m][0] - 5
            if h_prev_node < 0 and h_curr_node >= 0:
                switch_detected = True
                m_guess = m - 1
                break

        state_function = [u[m][0] - 5 for m in range(len(u))] if switch_detected else []
        return switch_detected, m_guess, state_function

    def count_switches(self):
        """
        Setter to update the number of switches if one is found.
        """
        self.nswitches += 1
