import numpy as np

from pySDC.core.Errors import ParameterError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.particles import particles, acceleration


# noinspection PyUnusedLocal
class henon_heiles(ptype):
    """
    Example implementing the harmonic oscillator
    """

    def __init__(self, problem_params, dtype_u=particles, dtype_f=acceleration):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: particle data type (will be passed parent class)
            dtype_f: acceleration data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        essential_keys = []
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # invoke super init, passing nparts, dtype_u and dtype_f
        super(henon_heiles, self).__init__((2, None, np.dtype('float64')), dtype_u, dtype_f, problem_params)

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
        me[0] = -u.pos[0] - 2 * u.pos[0] * u.pos[1]
        me[1] = -u.pos[1] - u.pos[0] ** 2 + u.pos[1] ** 2
        return me

    def u_exact(self, t):
        """
        Routine to compute the exact/initial trajectory at time t

        Args:
            t (float): current time
        Returns:
            dtype_u: exact/initial position and velocity
        """
        assert t == 0.0, 'error, u_exact only works for the initial time t0=0'
        me = self.dtype_u(self.init)

        q1 = 0.0
        q2 = 0.2
        p2 = 0.2
        U0 = 0.5 * (q1 * q1 + q2 * q2) + q1 * q1 * q2 - q2 * q2 * q2 / 3.0
        H0 = 0.125

        me.pos[0] = q1
        me.pos[1] = q2
        me.vel[0] = np.sqrt(2.0 * (H0 - U0) - p2 * p2)
        me.vel[1] = p2
        return me

    def eval_hamiltonian(self, u):
        """
        Routine to compute the Hamiltonian

        Args:
            u (dtype_u): the particles
        Returns:
            float: hamiltonian
        """

        ham = 0.5 * (u.vel[0] ** 2 + u.vel[1] ** 2)
        ham += 0.5 * (u.pos[0] ** 2 + u.pos[1] ** 2)
        ham += u.pos[0] ** 2 * u.pos[1] - u.pos[1] ** 3 / 3.0
        return ham
