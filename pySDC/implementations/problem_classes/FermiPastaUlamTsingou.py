import numpy as np

from pySDC.core.Errors import ParameterError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.particles import particles, acceleration


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
        super(fermi_pasta_ulam_tsingou, self).__init__(
            (problem_params['npart'], None, np.dtype('float64')), dtype_u, dtype_f, problem_params
        )

        self.dx = (self.params.npart / 32) / (self.params.npart + 1)
        self.xvalues = np.array([(i + 1) * self.dx for i in range(self.params.npart)])
        self.ones = np.ones(self.params.npart - 2)

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

        # me[1:-1] = u.pos[:-2] - 2.0 * u.pos[1:-1] + u.pos[2:] + \
        #     self.params.alpha * ((u.pos[2:] - u.pos[1:-1]) ** 2 -
        #                          (u.pos[1:-1] - u.pos[:-2]) ** 2)
        # me[0] = -2.0 * u.pos[0] + u.pos[1] + \
        #     self.params.alpha * ((u.pos[1] - u.pos[0]) ** 2 - (u.pos[0]) ** 2)
        # me[-1] = u.pos[-2] - 2.0 * u.pos[-1] + \
        #     self.params.alpha * ((u.pos[-1]) ** 2 - (u.pos[-1] - u.pos[-2]) ** 2)
        me[1:-1] = (u.pos[:-2] - 2.0 * u.pos[1:-1] + u.pos[2:]) * (
            self.ones + self.params.alpha * (u.pos[2:] - u.pos[:-2])
        )
        me[0] = (-2.0 * u.pos[0] + u.pos[1]) * (1 + self.params.alpha * (u.pos[1]))
        me[-1] = (u.pos[-2] - 2.0 * u.pos[-1]) * (1 + self.params.alpha * (-u.pos[-2]))

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

        me = self.dtype_u(self.init, val=0.0)

        me.pos[:] = np.sin(self.params.k * np.pi * self.xvalues)
        me.vel[:] = 0.0

        return me

    def eval_hamiltonian(self, u):
        """
        Routine to compute the Hamiltonian

        Args:
            u (dtype_u): the particles
        Returns:
            float: hamiltonian
        """

        ham = sum(
            0.5 * u.vel[:-1] ** 2
            + 0.5 * (u.pos[1:] - u.pos[:-1]) ** 2
            + self.params.alpha / 3.0 * (u.pos[1:] - u.pos[:-1]) ** 3
        )
        ham += 0.5 * u.vel[-1] ** 2 + 0.5 * (u.pos[-1]) ** 2 + self.params.alpha / 3.0 * (-u.pos[-1]) ** 3
        ham += 0.5 * (u.pos[0]) ** 2 + self.params.alpha / 3.0 * (u.pos[0]) ** 3
        return ham

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
            # Qk = np.sqrt(2.0 / (self.params.npart + 1)) * np.dot(u.pos, np.sin(np.pi * k * self.xvalues))
            Qk = np.sqrt(2.0 * self.dx) * np.dot(u.pos, np.sin(np.pi * k * self.xvalues))
            # Qkdot = np.sqrt(2.0 / (self.params.npart + 1)) * np.dot(u.vel, np.sin(np.pi * k * self.xvalues))
            Qkdot = np.sqrt(2.0 * self.dx) * np.dot(u.vel, np.sin(np.pi * k * self.xvalues))

            # omegak2 = 4.0 * np.sin(k * np.pi / (2.0 * (self.params.npart + 1))) ** 2
            omegak2 = 4.0 * np.sin(k * np.pi * self.dx / 2.0) ** 2
            energy[k] = 0.5 * (Qkdot**2 + omegak2 * Qk**2)

        return energy
