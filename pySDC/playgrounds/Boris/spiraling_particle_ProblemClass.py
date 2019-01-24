
import numpy as np

from pySDC.implementations.datatype_classes.particles import particles, fields, acceleration

from pySDC.core.Problem import ptype

class planewave_single(ptype):
    """
    Example implementing a single particle spiraling in a trap
    """

    def __init__(self, cparams, dtype_u=particles, dtype_f=fields):
        """
        Initialization routine

        Args:
            cparams: custom parameters for the example
            dtype_u: particle data type (will be passed parent class)
            dtype_f: fields data type (will be passed parent class)
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

        f = self.dtype_f((3, self.nparts))

        R = np.linalg.norm(part.pos.values[:, 0], 2)
        f.elec.values[0, 0] = self.params.a0 / (R ** 3) * part.pos.values[0, 0]
        f.elec.values[1, 0] = self.params.a0 / (R ** 3) * part.pos.values[1, 0]
        f.elec.values[2, 0] = 0

        f.magn.values[0, 0] = 0
        f.magn.values[1, 0] = 0
        f.magn.values[2, 0] = R

        return f

    def u_init(self):
        """
        Initialization routine for the single particle

        Returns:
            particle type
        """

        u0 = self.params.u0
        # some abbreviations
        u = self.dtype_u((3, 1))

        u.pos.values[0, 0] = u0[0][0]
        u.pos.values[1, 0] = u0[0][1]
        u.pos.values[2, 0] = u0[0][2]

        u.vel.values[0, 0] = u0[1][0]
        u.vel.values[1, 0] = u0[1][1]
        u.vel.values[2, 0] = u0[1][2]

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
        rhs = acceleration((3, self.nparts))
        rhs.values[:, 0] = part.q[:] / part.m[:] * \
            (f.elec.values[:, 0] + np.cross(part.vel.values[:, 0], f.magn.values[:, 0]))

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
        vel = particles.velocity((3, 1))

        Emean = 1.0 / 2.0 * (old_fields.elec + new_fields.elec)

        for n in range(N):
            a = old_parts.q[n] / old_parts.m[n]

            c.values[:, n] += dt / 2 * a * \
                np.cross(old_parts.vel.values[:, n], old_fields.magn.values[:, n] - new_fields.magn.values[:, n])

            # pre-velocity, separated by the electric forces (and the c term)
            vm = old_parts.vel.values[:, n] + dt / 2 * a * Emean.values[:, n] + c.values[:, n] / 2
            # rotation
            t = dt / 2 * a * new_fields.magn.values[:, n]
            s = 2 * t / (1 + np.linalg.norm(t, 2) ** 2)
            vp = vm + np.cross(vm + np.cross(vm, t), s)
            # post-velocity
            vel.values[:, n] = vp + dt / 2 * a * Emean.values[:, n] + c.values[:, n] / 2

        return vel
