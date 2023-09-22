import numpy as np

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class buck_converter(ptype):
    r"""
    Example implementing the model of a buck converter, which is also called a step-down converter. The converter has two different
    states and each of this state can be expressed as a nonhomogeneous linear system of ordinary differential equations (ODEs)

    .. math::
        \frac{d u(t)}{dt} = A_k u(t) + f_k (t)

    for :math:`k=1,2`. The two states are the following. Define :math:`T_{sw}:=\frac{1}{f_{sw}}` as the switching period with
    switching frequency :math:`f_{sw}`. The duty cycle :math:`d` defines the period of how long the switches are in one state
    until they switch to the other state. Roughly saying, the duty cycle can be seen as a percentage [1]_. A duty cycle of one means
    that the switches are always in only one state. If :math:`0 \leq \frac{t}{T_{sw}} \bmod 1 \leq d` [2]_:

    .. math::
        \frac{d v_{C_1} (t)}{dt} = -\frac{1}{R_s C_1}v_{C_1} (t) - \frac{1}{C_1} i_{L_1} (t) + \frac{V_s}{R_s C_1},

    .. math::
        \frac{d v_{C_2} (t)}{dt} = -\frac{1}{R_\ell C_2}v_{C_2} (t) + \frac{1}{C_2} i_{L_1} (t),

    .. math::
        \frac{d i_{L_1} (t)}{dt} = \frac{1}{L_1} v_{C_1} (t) - \frac{1}{L_1} v_{C_2} (t) - \frac{R_\pi}{L_1} i_{L_1} (t),

    Otherwise, the equations are

    .. math::
        \frac{d v_{C_1} (t)}{dt} = -\frac{1}{R_s C_1}v_{C_1} (t) + \frac{V_s}{R_s C_1},

    .. math::
        \frac{d v_{C_2} (t)}{dt} = -\frac{1}{R_\ell C_2}v_{C_2} (t) + \frac{1}{C_2} i_{L_1} (t),

    .. math::
        \frac{d i_{L_1} (t)}{dt} = \frac{R_\pi}{R_s L_1} v_{C_1} (t) - \frac{1}{L_1} v_{C_2} (t) -  \frac{R_\pi V_s}{L_1 R_s}.

    using an initial condition.

    Parameters
    ----------
    duty : float, optional
        Duty cycle :math:`d` between zero and one indicates the time period how long the converter stays on one switching
        state until it switches to the other state.
    fsw : int, optional
        Switching frequency, it is used to determine the number of time steps after the switching state is changed.
    Vs : float, optional
        Voltage at the voltage source :math:`V_s`.
    Rs : float, optional
        Resistance of the resistor :math:`R_s` at the voltage source.
    C1 : float, optional
        Capacitance of the capacitor :math:`C_1`.
    Rp : float, optional
        Resistance of the resistor in front of the inductor :math:`R_\pi`.
    L1 : float, optional
        Inductance of the inductor :math:`L_1`.
    C2 : float, optional
        Capacitance of the capacitor :math:`C_2`.
    Rl : float, optional
        Resistance of the resistor :math:`R_\pi`

    Attributes
    ----------
    A : np.2darray
        Coefficient matrix of the ODE system.

    Note
    ----
    The duty cycle needs to be a value in :math:`[0,1]`.

    References
    ----------
    .. [1] J. Sun. Pulse-Width Modulation. 25-61. Springer, (2012).
    .. [2] J. Gyselinck, C. Martis, R. V. Sabariego. Using dedicated time-domain basis functions for the simulation of
        pulse-width-modulation controlled devices - application to the steady-state regime of a buck converter. Electromotion (2013).
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(self, duty=0.5, fsw=1e3, Vs=10.0, Rs=0.5, C1=1e-3, Rp=0.01, L1=1e-3, C2=1e-3, Rl=10):
        """Initialization routine"""

        # invoke super init, passing number of dofs
        nvars = 3
        super().__init__(init=(nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'nvars', 'duty', 'fsw', 'Vs', 'Rs', 'C1', 'Rp', 'L1', 'C2', 'Rl', localVars=locals(), readOnly=True
        )

        self.A = np.zeros((nvars, nvars))

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
        Tsw = 1 / self.fsw

        f = self.dtype_f(self.init, val=0.0)
        f.impl[:] = self.A.dot(u)

        if 0 <= ((t / Tsw) % 1) <= self.duty:
            f.expl[0] = self.Vs / (self.Rs * self.C1)
            f.expl[2] = 0

        else:
            f.expl[0] = self.Vs / (self.Rs * self.C1)
            f.expl[2] = -(self.Rp * self.Vs) / (self.L1 * self.Rs)

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
        Tsw = 1 / self.fsw
        self.A = np.zeros((3, 3))

        if 0 <= ((t / Tsw) % 1) <= self.duty:
            self.A[0, 0] = -1 / (self.C1 * self.Rs)
            self.A[0, 2] = -1 / self.C1

            self.A[1, 1] = -1 / (self.C2 * self.Rl)
            self.A[1, 2] = 1 / self.C2

            self.A[2, 0] = 1 / self.L1
            self.A[2, 1] = -1 / self.L1
            self.A[2, 2] = -self.Rp / self.L1

        else:
            self.A[0, 0] = -1 / (self.C1 * self.Rs)

            self.A[1, 1] = -1 / (self.C2 * self.Rl)
            self.A[1, 2] = 1 / self.C2

            self.A[2, 0] = self.Rp / (self.L1 * self.Rs)
            self.A[2, 1] = -1 / self.L1

        me = self.dtype_u(self.init)
        me[:] = np.linalg.solve(np.eye(self.nvars) - factor * self.A, rhs)
        return me

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
        assert t == 0, 'ERROR: u_exact only valid for t=0'

        me = self.dtype_u(self.init)

        me[0] = 0.0  # v1
        me[1] = 0.0  # v2
        me[2] = 0.0  # p3

        return me
