from __future__ import division

import numpy as np

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.particles import particles, fields, acceleration
from pySDC.core.Errors import ParameterError, ProblemError


# noinspection PyUnusedLocal
class harmonic_oscillator(ptype):
    """
    Example implementing the harmonic oscillator
    """

    def __init__(self, problem_params, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: particle data type (will be passed parent class)
            dtype_f: acceleration data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        essential_keys = ['k', 'phase', 'amp']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # invoke super init, passing nparts, dtype_u and dtype_f
        super(harmonic_oscillator, self).__init__(1, dtype_u, dtype_f, problem_params)

    def eval_f(self, u, t):
        """
        Routine to compute the RHS

        Args:
            u (dtype_u): the particles
            t (float): current time (not used here)
        Returns:
            dtype_f: RHS
        """
        me = acceleration(1)
        me.values[:] = -self.params.k * u.pos.values
        return me

    def u_exact(self, t):
        """
        Routine to compute the exact trajectory at time t

        Args:
            t (float): current time
        Returns:
            dtype_u: exact position and velocity
        """

        me = particles(1)
        me.pos.values[:] = self.params.amp * np.cos(np.sqrt(self.params.k) * t + self.params.phase)
        me.vel.values[:] = -self.params.amp * np.sqrt(self.params.k) * np.sin(np.sqrt(self.params.k) * t +
                                                                              self.params.phase)
        return me

    def eval_hamiltonian(self, u):
        """
        Routine to compute the Hamiltonian

        Args:
            u (dtype_u): the particles
        Returns:
            float: hamiltonian
        """

        ham = 0.5 * self.params.k * u.pos.values[0] ** 2 + 0.5 * u.vel.values[0] ** 2
        return ham
