import numpy as np

from pySDC.core.Errors import ParameterError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.particles import particles, acceleration


# noinspection PyUnusedLocal
class fermi_pasta_ulam_tsingou(ptype):
    """
    Example implementing the outer solar system problem

    TODO : doku
    """

    dtype_u = particles
    dtype_f = acceleration

    def __init__(self, npart, alpha, k, energy_modes):
        """Initialization routine"""
        # invoke super init, passing nparts
        super().__init__((npart, None, np.dtype('float64')))
        self._makeAttributeAndRegister('npart', 'alpha', 'k', 'energy_modes', localVars=locals(), readOnly=True)

        self.dx = (self.npart / 32) / (self.npart + 1)
        self.xvalues = np.array([(i + 1) * self.dx for i in range(self.npart)])
        self.ones = np.ones(self.npart - 2)

    def eval_f(self, u, t):
        """
        Routine to compute the right-hand side of the problem.

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
        me = self.dtype_f(self.init, val=0.0)

        # me[1:-1] = u.pos[:-2] - 2.0 * u.pos[1:-1] + u.pos[2:] + \
        #     self.alpha * ((u.pos[2:] - u.pos[1:-1]) ** 2 -
        #                          (u.pos[1:-1] - u.pos[:-2]) ** 2)
        # me[0] = -2.0 * u.pos[0] + u.pos[1] + \
        #     self.alpha * ((u.pos[1] - u.pos[0]) ** 2 - (u.pos[0]) ** 2)
        # me[-1] = u.pos[-2] - 2.0 * u.pos[-1] + \
        #     self.alpha * ((u.pos[-1]) ** 2 - (u.pos[-1] - u.pos[-2]) ** 2)
        me[1:-1] = (u.pos[:-2] - 2.0 * u.pos[1:-1] + u.pos[2:]) * (self.ones + self.alpha * (u.pos[2:] - u.pos[:-2]))
        me[0] = (-2.0 * u.pos[0] + u.pos[1]) * (1 + self.alpha * (u.pos[1]))
        me[-1] = (u.pos[-2] - 2.0 * u.pos[-1]) * (1 + self.alpha * (-u.pos[-2]))

        return me

    def u_exact(self, t):
        """
        Routine to compute the exact/initial trajectory at time t.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            The exact/initial position and velocity.
        """
        assert t == 0.0, 'error, u_exact only works for the initial time t0=0'

        me = self.dtype_u(self.init, val=0.0)

        me.pos[:] = np.sin(self.k * np.pi * self.xvalues)
        me.vel[:] = 0.0

        return me

    def eval_hamiltonian(self, u):
        """
        Routine to compute the Hamiltonian.

        Parameters
        ----------
        u : dtype_u
            The particles.

        Returns
        -------
        ham : float
            The Hamiltonian.
        """

        ham = sum(
            0.5 * u.vel[:-1] ** 2
            + 0.5 * (u.pos[1:] - u.pos[:-1]) ** 2
            + self.alpha / 3.0 * (u.pos[1:] - u.pos[:-1]) ** 3
        )
        ham += 0.5 * u.vel[-1] ** 2 + 0.5 * (u.pos[-1]) ** 2 + self.alpha / 3.0 * (-u.pos[-1]) ** 3
        ham += 0.5 * (u.pos[0]) ** 2 + self.alpha / 3.0 * (u.pos[0]) ** 3
        return ham

    def eval_mode_energy(self, u):
        """
        Routine to compute the energy following
        http://www.scholarpedia.org/article/Fermi-Pasta-Ulam_nonlinear_lattice_oscillations

        Parameters
        ----------
        u : dtype_u
            The particles.

        Returns
        -------
        energy : dict
            The energies.
        """

        energy = {}

        for k in self.energy_modes:
            # Qk = np.sqrt(2.0 / (self.npart + 1)) * np.dot(u.pos, np.sin(np.pi * k * self.xvalues))
            Qk = np.sqrt(2.0 * self.dx) * np.dot(u.pos, np.sin(np.pi * k * self.xvalues))
            # Qkdot = np.sqrt(2.0 / (self.npart + 1)) * np.dot(u.vel, np.sin(np.pi * k * self.xvalues))
            Qkdot = np.sqrt(2.0 * self.dx) * np.dot(u.vel, np.sin(np.pi * k * self.xvalues))

            # omegak2 = 4.0 * np.sin(k * np.pi / (2.0 * (self.npart + 1))) ** 2
            omegak2 = 4.0 * np.sin(k * np.pi * self.dx / 2.0) ** 2
            energy[k] = 0.5 * (Qkdot**2 + omegak2 * Qk**2)

        return energy
