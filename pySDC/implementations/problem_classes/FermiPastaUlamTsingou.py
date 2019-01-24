from numba import jit
import numpy as np

from pySDC.implementations.datatype_classes.particles import particles, acceleration

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError


# noinspection PyUnusedLocal
class fermi_pasta_ulam_tsingou(ptype):
    """
    Example implementing the outer solar system problem
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
        essential_keys = ['npart', 'alpha', 'k', 'energy_modes']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # invoke super init, passing nparts, dtype_u and dtype_f
        super(fermi_pasta_ulam_tsingou, self).__init__(problem_params['npart'], dtype_u, dtype_f, problem_params)

    @staticmethod
    @jit
    def fast_acceleration(N, alpha, pos, accel):
        for n in range(1, N - 1):
            accel[n - 1] = pos[n - 1] - 2.0 * pos[n] + pos[n + 1] + \
                alpha * ((pos[n + 1] - pos[n]) ** 2 - (pos[n] - pos[n - 1]) ** 2)

    def eval_f(self, u, t):
        """
        Routine to compute the RHS

        Args:
            u (dtype_u): the particles
            t (float): current time (not used here)
        Returns:
            dtype_f: RHS
        """
        me = self.dtype_f(self.init, val=0.0)

        self.fast_acceleration(self.params.npart, self.params.alpha, u.pos.values, me.values[1:self.params.npart - 1])

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

        for n in range(self.params.npart):
            me.pos.values[n] = np.sin(self.params.k * np.pi * n / (self.params.npart - 1))
            me.vel.values[n] = 0.0

        return me

    @staticmethod
    @jit
    def fast_hamiltonian(N, alpha, pos, vel):
        ham = 0.0
        for n in range(N - 1):
            ham += 0.5 * vel[n] ** 2 + 0.5 * (pos[n + 1] - pos[n]) ** 2 + alpha / 3.0 * (pos[n + 1] - pos[n]) ** 3
        return ham

    def eval_hamiltonian(self, u):
        """
        Routine to compute the Hamiltonian

        Args:
            u (dtype_u): the particles
        Returns:
            float: hamiltonian
        """
        return self.fast_hamiltonian(self.params.npart, self.params.alpha, u.pos.values, u.vel.values)

    def eval_mode_energy(self, u):
        """
        Routine to compute the energy following
        http://www.scholarpedia.org/article/Fermi-Pasta-Ulam_nonlinear_lattice_oscillations

        Args:
            u (dtype_u): the particles
        Returns:
            dict: energies
        """

        energy = {}

        for k in self.params.energy_modes:

            Qk = np.sqrt(2.0 / (self.params.npart - 1)) * \
                sum([u.pos.values[n] * np.sin(np.pi * k * n / (self.params.npart - 1))
                     for n in range(1, self.params.npart)])
            Qkdot = np.sqrt(2.0 / (self.params.npart - 1)) * \
                sum([u.vel.values[n] * np.sin(np.pi * k * n / (self.params.npart - 1))
                     for n in range(1, self.params.npart)])

            omegak2 = 4.0 * np.sin(k * np.pi / (2.0 * (self.params.npart - 1))) ** 2
            energy[k] = 0.5 * (Qkdot ** 2 + omegak2 * Qk ** 2)

        return energy
