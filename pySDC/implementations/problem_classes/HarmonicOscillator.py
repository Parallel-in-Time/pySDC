import numpy as np

from pySDC.core.Errors import ParameterError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.particles import particles, acceleration


# noinspection PyUnusedLocal
class harmonic_oscillator(ptype):
    """
    Example implementing the harmonic oscillator
    """

    def __init__(self, problem_params, dtype_u=particles, dtype_f=acceleration):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: particle data type (will be passed to parent class)
            dtype_f: acceleration data type (will be passed to parent class)
        """

        # these parameters will be used later, so assert their existence
        problem_params["phase"] = 1.0
        problem_params["amp"] = 0.0
        essential_keys = ["k", "mu", "u0", "phase", "amp"]

        for key in essential_keys:
            if key == "mu" and key not in problem_params:
                problem_params["mu"] = 0.0
            if key == "u0" and key not in problem_params:
                problem_params["u0"] = np.array([1, 0])
            elif key not in problem_params:
                msg = "need %s to instantiate problem, only got %s" % (
                    key,
                    str(problem_params.keys()),
                )
                raise ParameterError(msg)

        # invoke super init, passing nparts, dtype_u and dtype_f
        super(harmonic_oscillator, self).__init__((1, None, np.dtype("float64")), dtype_u, dtype_f, problem_params)

    def eval_f(self, u, t):
        """
        Routine to compute the RHS

        Args:
            u (dtype_u): the particles
            t (float): current time (not used here)
        Returns:
            dtype_f: RHS
        """
        me = self.dtype_f(self.init)
        me[:] = -self.params.k * u.pos - self.params.mu * u.vel
        return me

    def u_init(self):

        u0 = self.params.u0

        u = self.dtype_u(self.init)

        u.pos[0] = u0[0]
        u.vel[0] = u0[1]

        return u

    def u_exact(self, t):
        """
        Routine to compute the exact trajectory at time t

        Args:
            t (float): current time
        Returns:
            dtype_u: exact position and velocity
        """
        me = self.dtype_u(self.init)
        delta = self.params.mu / (2)
        omega = np.sqrt(self.params.k)

        U_0 = self.params.u0
        alpha = np.sqrt(np.abs(delta**2 - omega**2))
        print(self.params.mu)
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
        Routine to compute the Hamiltonian

        Args:
            u (dtype_u): the particles
        Returns:
            float: hamiltonian
        """

        ham = 0.5 * self.params.k * u.pos[0] ** 2 + 0.5 * u.vel[0] ** 2
        return ham
