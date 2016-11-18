from __future__ import division

import numpy as np

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.particles import particles, fields, acceleration
from pySDC.core.Errors import ParameterError, ProblemError


# noinspection PyUnusedLocal
class penningtrap(ptype):
    """
    Example implementing particles in a penning trap
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
        essential_keys = ['omega_B', 'omega_E', 'u0', 'nparts', 'sig']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # invoke super init, passing nparts, dtype_u and dtype_f
        super(penningtrap, self).__init__(problem_params['nparts'], dtype_u, dtype_f, problem_params)

    def get_interactions(self, part):
        """
        Routine to compute the particle-particle interaction, assuming q = 1 for all particles

        Args:
            part (dtype_u): the particles
        Returns:
            numpy.ndarray: the internal E field for each particle

        """

        N = self.params.nparts

        Efield = np.zeros(3 * N)

        for i in range(N):
            for j in range(N):
                dist2 = np.linalg.norm(part.pos.values[3 * i:3 * i + 3] - part.pos.values[3 * j:3 * j + 3], 2) ** 2 \
                    + self.params.sig ** 2
                contrib = part.q[j] * (part.pos.values[3 * i:3 * i + 3] - part.pos.values[3 * j:3 * j + 3]) / \
                    dist2 ** (3 / 2)

                Efield[3 * i] += contrib[0]
                Efield[3 * i + 1] += contrib[1]
                Efield[3 * i + 2] += contrib[2]

        return Efield

    def eval_f(self, part, t):
        """
        Routine to compute the E and B fields (named f for consistency with the original PEPC version)

        Args:
            part (dtype_u): the particles
            t (float): current time (not used here)
        Returns:
            dtype_f: Fields for the particles (internal and external)
        """

        N = self.params.nparts

        Emat = np.diag([1, 1, -2])
        f = fields(self.params.nparts)

        f.elec.values = self.get_interactions(part)

        for n in range(N):
            f.elec.values[3 * n:3 * n + 3] += self.params.omega_E ** 2 / (part.q[n] / part.m[n]) * \
                                              np.dot(Emat, part.pos.values[3 * n:3 * n + 3])
            f.magn.values[3 * n:3 * n + 3] = self.params.omega_B * np.array([0, 0, 1])

        return f

    def u_init(self):
        """
        Routine to compute the starting values for the particles

        Returns:
            dtype_u: particle set filled with initial data
        """

        u0 = self.params.u0
        N = self.params.nparts

        u = particles(N)

        if u0[2][0] is not 1 or u0[3][0] is not 1:
            raise ProblemError('so far only q = m = 1 is implemented')

        # set first particle to u0
        u.pos.values[0] = u0[0][0]
        u.pos.values[1] = u0[0][1]
        u.pos.values[2] = u0[0][2]
        u.vel.values[0] = u0[1][0]
        u.vel.values[1] = u0[1][1]
        u.vel.values[2] = u0[1][2]

        u.q[0] = u0[2][0]
        u.m[0] = u0[3][0]

        # initialize random seed
        np.random.seed(N)

        comx = u.pos.values[0]
        comy = u.pos.values[1]
        comz = u.pos.values[2]

        for n in range(1, N):
            # draw 3 random variables in [-1,1] to shift positions
            r = np.random.random_sample(3) - 1
            u.pos.values[3 * n] = r[0] + u0[0][0]
            u.pos.values[3 * n + 1] = r[1] + u0[0][1]
            u.pos.values[3 * n + 2] = r[2] + u0[0][2]

            # draw 3 random variables in [-5,5] to shift velocities
            r = np.random.random_sample(3) - 5
            u.vel.values[3 * n] = r[0] + u0[1][0]
            u.vel.values[3 * n + 1] = r[1] + u0[1][1]
            u.vel.values[3 * n + 2] = r[2] + u0[1][2]

            u.q[n] = u0[2][0]
            u.m[n] = u0[3][0]

            # gather positions to check center
            comx += u.pos.values[3 * n]
            comy += u.pos.values[3 * n + 1]
            comz += u.pos.values[3 * n + 2]

        # print('Center of positions:',comx/N,comy/N,comz/N)

        return u

    def u_exact(self, t):
        """
        Routine to compute the exact trajectory at time t (only for single-particle setup)

        Args:
            t (float): current time
        Returns:
            dtype_u: particle type containing the exact position and velocity
        """

        # some abbreviations
        wE = self.params.omega_E
        wB = self.params.omega_B
        N = self.params.nparts
        u0 = self.params.u0

        if N != 1:
            raise ProblemError('u_exact is only valid for a single particle')

        u = particles(1)

        wbar = np.sqrt(2) * wE

        # position and velocity in z direction is easy to compute
        u.pos.values[2] = u0[0][2] * np.cos(wbar * t) + u0[1][2] / wbar * np.sin(wbar * t)
        u.vel.values[2] = -u0[0][2] * wbar * np.sin(wbar * t) + u0[1][2] * np.cos(wbar * t)

        # define temp. variables to compute complex position
        Op = 1 / 2 * (wB + np.sqrt(wB ** 2 - 4 * wE ** 2))
        Om = 1 / 2 * (wB - np.sqrt(wB ** 2 - 4 * wE ** 2))
        Rm = (Op * u0[0][0] + u0[1][1]) / (Op - Om)
        Rp = u0[0][0] - Rm
        Im = (Op * u0[0][1] - u0[1][0]) / (Op - Om)
        Ip = u0[0][1] - Im

        # compute position in complex notation
        w = np.complex(Rp, Ip) * np.exp(-np.complex(0, Op * t)) + np.complex(Rm, Im) * np.exp(-np.complex(0, Om * t))
        # compute velocity as time derivative of the position
        dw = -1j * Op * np.complex(Rp, Ip) * \
            np.exp(-np.complex(0, Op * t)) - 1j * Om * np.complex(Rm, Im) * np.exp(-np.complex(0, Om * t))

        # get the appropriate real and imaginary parts
        u.pos.values[0] = w.real
        u.vel.values[0] = dw.real
        u.pos.values[1] = w.imag
        u.vel.values[1] = dw.imag

        return u

    def build_f(self, f, part, t):
        """
        Helper function to assemble the correct right-hand side out of B and E field

        Args:
            f (dtype_f): the field values
            part (dtype_u): the current particles data
            t (float): the current time
        Returns:
            acceleration: correct RHS of type acceleration
        """

        if not isinstance(part, particles):
            raise ProblemError('something is wrong during build_f, got %s' % type(part))

        N = self.params.nparts

        rhs = acceleration(self.params.nparts)

        for n in range(N):
            rhs.values[3 * n:3 * n + 3] = part.q[n] / part.m[n] * (
                f.elec.values[3 * n:3 * n + 3] +
                np.cross(part.vel.values[3 * n:3 * n + 3], f.magn.values[3 * n:3 * n + 3]))

        return rhs

    # noinspection PyTypeChecker
    def boris_solver(self, c, dt, old_fields, new_fields, old_parts):
        """
        The actual Boris solver for static (!) B fields, extended by the c-term

        Args:
            c (dtype_u): the c term gathering the known values from the previous iteration
            dt (float): the (probably scaled) time step size
            old_fields (dtype_f): the field values at the previous node m
            new_fields (dtype_f): the field values at the current node m+1
            old_parts (dtype_u): the particles at the previous node m
        Returns:
            the velocities at the (m+1)th node
        """

        N = self.params.nparts
        vel = particles.velocity(N)

        Emean = 0.5 * (old_fields.elec + new_fields.elec)

        for n in range(N):
            a = old_parts.q[n] / old_parts.m[n]

            c.values[3 * n:3 * n + 3] += dt / 2 * a * \
                np.cross(old_parts.vel.values[3 * n:3 * n + 3], old_fields.magn.values[3 * n:3 * n + 3] -
                         new_fields.magn.values[3 * n:3 * n + 3])

            # pre-velocity, separated by the electric forces (and the c term)
            vm = old_parts.vel.values[3 * n:3 * n + 3] + dt / 2 * a * Emean.values[3 * n:3 * n + 3] + \
                c.values[3 * n:3 * n + 3] / 2
            # rotation
            t = dt / 2 * a * new_fields.magn.values[3 * n:3 * n + 3]
            s = 2 * t / (1 + np.linalg.norm(t, 2) ** 2)
            vp = vm + np.cross(vm + np.cross(vm, t), s)
            # post-velocity
            vel.values[3 * n:3 * n + 3] = vp + dt / 2 * a * Emean.values[3 * n:3 * n + 3] + \
                c.values[3 * n:3 * n + 3] / 2

        return vel
