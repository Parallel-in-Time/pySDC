from __future__ import division

import numpy as np

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.particles import particles, fields, acceleration


class planewave_single(ptype):
    """
    Example implementing a single particle spiraling in a trap
    """

    def __init__(self, cparams, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            cparams: custom parameters for the example
            dtype_u: particle data type (will be passed parent class)
            dtype_f: acceleration data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        assert 'delta' in cparams  # polarization
        assert 'a0' in cparams  # normalized amplitude
        assert 'u0' in cparams  # initial position and velocity

        # add parameters as attributes for further reference
        for k, v in cparams.items():
            setattr(self, k, v)

        # set nparts to one (lonely particle, you know)
        self.nparts = 1
        # invoke super init, passing nparts, dtype_u and dtype_f
        super(planewave_single, self).__init__(self.nparts, dtype_u, dtype_f, cparams)

    def eval_f(self, part, t):
        """
        Routine to compute the electric and magnetic fields

        Args:
            t: current time
            part: the current particle
        Returns:
            E and B field for the particle (external only)
        """

        f = fields(self.nparts)

        R = np.linalg.norm(part.pos.values, 2)
        f.elec.values[0] = self.params.a0 / (R ** 3) * part.pos.values[0]
        f.elec.values[1] = self.params.a0 / (R ** 3) * part.pos.values[1]
        f.elec.values[2] = 0

        f.magn.values[0] = 0
        f.magn.values[1] = 0
        f.magn.values[2] = R

        return f

    def u_init(self):
        """
        Initialization routine for the single particle

        Returns:
            particle type
        """

        u0 = self.u0
        # some abbreviations
        u = particles(1)

        u.pos.values[0] = u0[0][0]
        u.pos.values[1] = u0[0][1]
        u.pos.values[2] = u0[0][2]

        u.vel.values[0] = u0[1][0]
        u.vel.values[1] = u0[1][1]
        u.vel.values[2] = u0[1][2]

        u.q[:] = u0[2][0]
        u.m[:] = u0[3][0]

        return u

    def build_f(self, f, part, t):
        """
        Helper function to assemble the correct right-hand side out of B and E field

        Args:
            f: wannabe right-hand side, actually the E field
            part: particle data
            t: current time
        Returns:
            correct RHS of type acceleration
        """

        assert isinstance(part, particles)
        rhs = acceleration(self.nparts)
        rhs.values[:] = part.q[:] / part.m[:] * (f.elec.values + np.cross(part.vel.values, f.magn.values))

        return rhs

    def boris_solver(self, c, dt, old_fields, new_fields, old_parts):
        """
        The actual Boris solver for static (!) B fields, extended by the c-term

        Args:
            c: the c term gathering the known values from the previous iteration
            dt: the (probably scaled) time step size
            old_fields: the field values at the previous node m
            new_fields: the field values at the current node m+1
            old_parts: the particles at the previous node m
        Returns:
            the velocities at the (m+1)th node
        """

        N = self.nparts
        vel = particles.velocity(N)

        Emean = 1.0 / 2.0 * (old_fields.elec + new_fields.elec)

        for n in range(N):
            a = old_parts.q[n] / old_parts.m[n]

            c.values[3 * n:3 * n + 3] += dt / 2 * a * np.cross(old_parts.vel.values[3 * n:3 * n + 3],
                                                               old_fields.magn.values[3 * n:3 * n + 3] -
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
