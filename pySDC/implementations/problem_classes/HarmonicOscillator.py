import numpy as np

from pySDC.core.Errors import ParameterError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.particles import particles, acceleration


# noinspection PyUnusedLocal
class harmonic_oscillator(ptype):
    r"""
    Example implementing the harmonic oscillator with mass :math:`1`

    .. math::
        \frac{d^2 x}{dt^2} = -kx - \mu \frac{d x}{dt},

    which is a second-order problem. The unknown function :math:`x` denotes the position of the mass, and their
    derivative is the velocity. :math:`\mu` defines the damping and :math:`k` is the spring constant.

    Parameters
    ----------
    k : float, optional
        Spring constant :math:`k`.
    mu : float, optional
        Damping parameter :math:`\mu`.
    u0 : tuple, optional
        Initial condition for the position, and the velocity. Should be a tuple, e.g. (1, 0).
    phase : float, optional
        Phase of the oscillation.
    amp : float, optional
        Amplitude of the oscillation.
    """

    dtype_u = particles
    dtype_f = acceleration

    def __init__(self, k=1.0, mu=0.0, u0=(1, 0), phase=0.0, amp=1.0):
        """Initialization routine"""
        # invoke super init, passing nparts, dtype_u and dtype_f
        u0 = np.asarray(u0)
        super().__init__((1, None, np.dtype("float64")))
        self._makeAttributeAndRegister('k', 'mu', 'u0', 'phase', 'amp', localVars=locals(), readOnly=True)

    def eval_f(self, u, t):
        """
        Routine to compute the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the particles.
        t : float
            Current time of the numerical solution is computed (not used here).

        Returns
        -------
        me : dtype_f
            The right-hand side of the problem.
        """
        me = self.dtype_f(self.init)
        me[:] = -self.k * u.pos - self.mu * u.vel
        return me

    def u_init(self):
        """
        Helper function to compute the initial condition for u.
        """
        u0 = self.u0

        u = self.dtype_u(self.init)

        u.pos[0] = u0[0]
        u.vel[0] = u0[1]

        return u

    def u_exact(self, t):
        """
        Routine to compute the exact trajectory at time t.

        Parameters
        ----------
        t : float
            Time of the exact trajectory.

        Returns
        -------
        me : dtype_u
            Exact position and velocity.
        """
        me = self.dtype_u(self.init)
        delta = self.mu / (2)
        omega = np.sqrt(self.k)

        U_0 = self.u0
        alpha = np.sqrt(np.abs(delta**2 - omega**2))
        print(self.mu)
        if delta > omega:
            """
            Overdamped case
            """

            lam_1 = -delta + alpha
            lam_2 = -delta - alpha
            L = np.array([[1, 1], [lam_1, lam_2]])
            A, B = np.linalg.solve(L, U_0)
            me.pos[:] = A * np.exp(lam_1 * t) + B * np.exp(lam_2 * t)
            me.vel[:] = A * lam_1 * np.exp(lam_1 * t) + B * lam_2 * np.exp(lam_2 * t)

        elif delta == omega:
            """
            Critically damped case
            """

            A = U_0[0]
            B = U_0[1] + delta * A
            me.pos[:] = np.exp(-delta * t) * (A + t * B)
            me.vel[:] = -delta * me.pos[:] + np.exp(-delta * t) * B

        elif delta < omega:
            """
            Underdamped case
            """

            lam_1 = -delta + alpha * 1j
            lam_2 = -delta - alpha * 1j

            M = np.array([[1, 1], [lam_1, lam_2]], dtype=complex)
            A, B = np.linalg.solve(M, U_0)

            me.pos[:] = np.real(A * np.exp(lam_1 * t) + B * np.exp(lam_2 * t))
            me.vel[:] = np.real(A * lam_1 * np.exp(lam_1 * t) + B * lam_2 * np.exp(lam_2 * t))

        else:
            pass
            raise ParameterError("Exact solution is not working")
        return me

    def eval_hamiltonian(self, u):
        """
        Routine to compute the Hamiltonian.

        Parameters
        ----------
        u : dtype_u
            Current values of the particles.

        Returns
        -------
        ham : float
            The Hamiltonian.
        """

        ham = 0.5 * self.k * u.pos[0] ** 2 + 0.5 * u.vel[0] ** 2
        return ham
