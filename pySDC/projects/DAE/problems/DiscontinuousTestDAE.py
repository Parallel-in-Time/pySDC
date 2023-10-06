import numpy as np

from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae


class DiscontinuousTestDAE(ptype_dae):
    r"""
    This class implements a scalar test discontinuous differential-algebraic equation (DAE) similar to [1]_. The event function
    is defined as :math:`h(y):= 2y - 100`. Then, the discontinuous DAE model reads:

    - if :math:`h(y) \leq 0`:

        .. math::
            \dfrac{d y(t)}{dt} = z(t),

        .. math::
            y(t)^2 - z(t)^2 - 1 = 0,

    else:

        .. math::
            \dfrac{d y(t)}{dt} = 0,

        .. math::
            y(t)^2 - z(t)^2 - 1 = 0,

    for :math:`t \geq 1`. If :math:`h(y) < 0`, the solution is given by

    .. math::
        (y(t), z(t)) = (cosh(t), sinh(t)),

    and the event takes place at :math:`t^* = 0.5 * arccosh(50) = 4.60507` and event point :math:`(cosh(t^*), sinh(t^*))`.
    As soon as :math:`t \geq t^*`, i.e., for :math:`h(y) \geq 0`, the solution is given by the constant value
    :math:`(cosh(t^*), sinh(t^*))`.

    Parameters
    ----------
    nvars : int
        Number of unknowns of the system of DAEs.
    newton_tol : float
        Tolerance for Newton solver.

    Attributes
    ----------
    t_switch_exact: float
        Exact time of the event :math:`t^*`.
    t_switch: float
        Time point of the event found by switch estimation.
    nswitches: int
        Number of switches found by switch estimation.

    References
    ----------
    .. [1] L. Lopez, S. Maset. Numerical event location techniques in discontinuous differential algebraic equations.
        Appl. Numer. Math. 178, 98-122 (2022).
    """

    def __init__(self, newton_tol=1e-12):
        """Initialization routine"""
        nvars = 2
        super().__init__(nvars, newton_tol)
        self._makeAttributeAndRegister('nvars', localVars=locals(), readOnly=True)
        self._makeAttributeAndRegister('newton_tol', localVars=locals())

        self.t_switch_exact = np.arccosh(50)
        self.t_switch = None
        self.nswitches = 0

    def eval_f(self, u, du, t):
        r"""
        Routine to evaluate the implicit representation of the problem, i.e., :math:`F(u, u', t)`.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution at time t.
        du : dtype_u
            Current values of the derivative of the numerical solution at time t.
        t : float
            Current time of the numerical solution.

        Returns
        -------
        f : dtype_f
            The right-hand side of f (contains two components).
        """

        y, z = u[0], u[1]
        dy = du[0]

        t_switch = np.inf if self.t_switch is None else self.t_switch

        h = 2 * y - 100
        f = self.dtype_f(self.init)

        f_before_event = (
            dy - z,
            y**2 - z**2 - 1,
        )

        f_after_event = (
            dy,
            y**2 - z**2 - 1,
        )

        if h >= 0 or t >= t_switch:
            f[:] = f_after_event
        else:
            f[:] = f_before_event
        return f

    def u_exact(self, t, u_init=None, t_init=None):
        r"""
        Routine for the exact solution at time :math:`t \leq 1`. For this problem, the exact
        solution is piecewise.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            Exact solution.
        """

        assert t >= 1, 'ERROR: u_exact only valid for t>=1'

        me = self.dtype_u(self.init)
        if t <= self.t_switch_exact:
            me[:] = (np.cosh(t), np.sinh(t))
        else:
            me[:] = (np.cosh(self.t_switch_exact), np.sinh(self.t_switch_exact))
        return me

    def get_switching_info(self, u, t):
        r"""
        Provides information about the state function of the problem. When the state function changes its sign,
        typically an event occurs. So the check for an event should be done in the way that the state function
        is checked for a sign change. If this is the case, the intermediate value theorem states a root in this
        step.

        The state function for this problem is given by

        .. math::
           h(y(t)) = 2 y(t) - 100.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution at time :math:`t`.
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
            h_prev_node = 2 * u[m - 1][0] - 100
            h_curr_node = 2 * u[m][0] - 100
            if h_prev_node < 0 and h_curr_node >= 0:
                switch_detected = True
                m_guess = m - 1
                break

        state_function = [2 * u[m][0] - 100 for m in range(len(u))]
        return switch_detected, m_guess, state_function

    def count_switches(self):
        """
        Setter to update the number of switches if one is found.
        """
        self.nswitches += 1