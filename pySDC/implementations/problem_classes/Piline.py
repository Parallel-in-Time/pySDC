import numpy as np
from scipy.integrate import solve_ivp

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


# noinspection PyUnusedLocal
class piline(ptype):
    r"""
    Example implementing the model of the piline. It serves as a transmission line in an energy grid. The problem of simulating the
    piline consists of three ordinary differential equations (ODEs) with nonhomogeneous part:

    .. math::
        \frac{d v_{C_1} (t)}{dt} = -\frac{1}{R_s C_1}v_{C_1} (t) - \frac{1}{C_1} i_{L_\pi} (t) + \frac{V_s}{R_s C_1},

    .. math::
        \frac{d v_{C_2} (t)}{dt} = -\frac{1}{R_\ell C_2}v_{C_2} (t) + \frac{1}{C_2} i_{L_\pi} (t),

    .. math::
        \frac{d i_{L_\pi} (t)}{dt} = \frac{1}{L_\pi} v_{C_1} (t) - \frac{1}{L_\pi} v_{C_2} (t) - \frac{R_\pi}{L_\pi} i_{L_\pi} (t),

    which can be expressed as a nonhomogeneous linear system of ODEs

    .. math::
        \frac{d u(t)}{dt} = A u(t) + f(t)

    using an initial condition.

    Parameters
    ----------
    Vs : float
        Voltage at the voltage source :math:`V_s`.
    Rs : float
        Resistance of the resistor :math:`R_s` at the voltage source.
    C1 : float
        Capacitance of the capacitor :math:`C_1`.
    Rpi : float
        Resistance of the resistor :math:`R_\pi`.
    Lpi : float
        Inductance of the inductor :math:`L_\pi`.
    C2 : float
        Capacitance of the capacitor :math:`C_2`.
    Rl : float
        Resistance of the resistive load :math:`R_\ell`.

    Attributes:
        A: system matrix, representing the 3 ODEs
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(self, Vs, Rs, C1, Rpi, Lpi, C2, Rl):
        """Initialization routine"""

        nvars = 3
        # invoke super init, passing number of dofs
        super().__init__(init=(nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'nvars', 'Vs', 'Rs', 'C1', 'Rpi', 'Lpi', 'C2', 'Rl', localVars=locals(), readOnly=True
        )

        # compute dx and get discretization matrix A
        self.A = np.zeros((3, 3))
        self.A[0, 0] = -1 / (self.Rs * self.C1)
        self.A[0, 2] = -1 / self.C1
        self.A[1, 1] = -1 / (self.Rl * self.C2)
        self.A[1, 2] = 1 / self.C2
        self.A[2, 0] = 1 / self.Lpi
        self.A[2, 1] = -1 / self.Lpi
        self.A[2, 2] = -self.Rpi / self.Lpi

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

        f = self.dtype_f(self.init, val=0.0)
        f.impl[:] = self.A.dot(u)
        f.expl[0] = self.Vs / (self.Rs * self.C1)
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

        me = self.dtype_u(self.init)
        me[:] = np.linalg.solve(np.eye(self.nvars) - factor * self.A, rhs)
        return me

    def u_exact(self, t, u_init=None, t_init=None):
        """
        Routine to approximate the exact solution at time t by scipy as a reference.

        Parameters
        ----------
        t : float
            Time of the exact solution.
        u_init : pySDC.problem.Piline.dtype_u
            Initial conditions for getting the exact solution.
        t_init : float
            The starting time.

        Returns
        -------
        me : dtype_u
            The reference solution.
        """

        me = self.dtype_u(self.init)

        # fill initial conditions
        me[0] = 0.0  # v1
        me[1] = 0.0  # v2
        me[2] = 0.0  # p3

        if t > 0.0:
            if u_init is not None:
                if t_init is None:
                    raise ValueError(
                        'Please supply `t_init` when you want to get the exact solution from a point that \
is not 0!'
                    )
                me = u_init
            else:
                t_init = 0.0

            def rhs(t, u):
                f = self.eval_f(u, t)
                return f.impl + f.expl  # evaluate only explicitly rather than IMEX

            tol = 100 * np.finfo(float).eps

            me[:] = solve_ivp(rhs, (t_init, t), me, rtol=tol, atol=tol).y[:, -1]

        return me
