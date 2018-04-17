from __future__ import division

import numpy as np

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.particles import particles, fields, acceleration
from pySDC.core.Errors import ParameterError, ProblemError


# noinspection PyUnusedLocal
class outer_solar_system(ptype):
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
        essential_keys = []
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # invoke super init, passing nparts, dtype_u and dtype_f
        super(outer_solar_system, self).__init__((3, 6), dtype_u, dtype_f, problem_params)
        self.G = 2.95912208286E-4

    def eval_f(self, u, t):
        """
        Routine to compute the RHS

        Args:
            u (dtype_u): the particles
            t (float): current time (not used here)
        Returns:
            dtype_f: RHS
        """
        me = acceleration(self.init, val=0.0)

        for i in range(self.init[-1]):
            for j in range(i):
                dx = u.pos.values[:, i] - u.pos.values[:, j]
                r = np.sqrt(np.dot(dx, dx))
                df = self.G * dx / (r ** 3)
                me.values[:, i] -= u.m[j] * df
                me.values[:, j] += u.m[i] * df

        return me

    def u_exact(self, t):
        """
        Routine to compute the exact trajectory at time t

        Args:
            t (float): current time
        Returns:
            dtype_u: exact position and velocity
        """
        assert t == 0.0, 'error, u_exact only works for the initial time t0=0'
        me = particles(self.init)

        me.pos.values[:, 0] = [0.0, 0.0, 0.0]
        me.pos.values[:, 1] = [-3.5025653, -3.8169847, -1.5507963]
        me.pos.values[:, 2] = [9.0755314, -3.0458353, -1.6483708]
        me.pos.values[:, 3] = [8.3101420, -16.2901086, -7.2521278]
        me.pos.values[:, 4] = [11.4707666, -25.7294829, -10.8169456]
        me.pos.values[:, 5] = [-15.5387357, -25.2225594, -3.1902382]

        me.vel.values[:, 0] = [0.0, 0.0, 0.0]
        me.vel.values[:, 1] = [0.00565429, -0.00412490, -0.00190589]
        me.vel.values[:, 2] = [0.00168318, 0.00483525, 0.00192462]
        me.vel.values[:, 3] = [0.00354178, 0.00137102, 0.00055029]
        me.vel.values[:, 4] = [0.00288930, 0.00114527, 0.00039677]
        me.vel.values[:, 5] = [0.00276725, -0.0017072, -0.00136504]

        me.m[0] = 1.00000597682
        me.m[1] = 0.000954786104043
        me.m[2] = 0.000285583733151
        me.m[3] = 0.0000437273164546
        me.m[4] = 0.0000517759138449
        me.m[5] = 1.0 / 130000000.0

        return me

    def eval_hamiltonian(self, u):
        """
        Routine to compute the Hamiltonian

        Args:
            u (dtype_u): the particles
        Returns:
            float: hamiltonian
        """

        ham = 0

        for i in range(self.init[-1]):
            ham += 0.5 * u.m[i] * np.dot(u.vel.values[:, i], u.vel.values[:, i])

        for i in range(self.init[-1]):
            for j in range(i):
                dx = u.pos.values[:, i] - u.pos.values[:, j]
                r = np.sqrt(np.dot(dx, dx))
                ham -= self.G * u.m[i] * u.m[j] / r

        return ham
