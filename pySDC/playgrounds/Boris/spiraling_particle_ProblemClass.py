import numpy as np

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.particles import particles, fields, acceleration


class planewave_single(ptype):
    """
    Example implementing a single particle spiraling in a trap
    """
    dtype_u = particles
    dtype_f = fields

    def __init__(self, u0, a0, delta):
        """
        Initialization routine

        Args:
            cparams: custom parameters for the example
            dtype_u: particle data type (will be passed parent class)
            dtype_f: fields data type (will be passed parent class)
        """
        # set nparts to one (lonely particle, you know)
        nparts = 1
        # TODO : delta is not used later in the Problem class or Hook class !

        super().__init__(((3, nparts), None, np.dtype('float64')))
        self._makeAttributeAndRegister('nparts', 'a0', 'delta', localVars=locals(), readOnly=True)
        self._makeAttributeAndRegister('u0', localVars=locals())

    def eval_f(self, part, t):
        """
        Routine to compute the electric and magnetic fields

        Args:
            t: current time
            part: the current particle
        Returns:
            E and B field for the particle (external only)
        """

        f = self.dtype_f(((3, self.nparts), self.init[1], self.init[2]))

        R = np.linalg.norm(part.pos[:, 0], 2)
        f.elec[0, 0] = self.a0 / (R**3) * part.pos[0, 0]
        f.elec[1, 0] = self.a0 / (R**3) * part.pos[1, 0]
        f.elec[2, 0] = 0

        f.magn[0, 0] = 0
        f.magn[1, 0] = 0
        f.magn[2, 0] = R

        return f

    def u_init(self):
        """
        Initialization routine for the single particle

        Returns:
            particle type
        """

        u0 = self.u0
        # some abbreviations
        u = self.dtype_u(((3, 1), self.init[1], self.init[2]))

        u.pos[0, 0] = u0[0][0]
        u.pos[1, 0] = u0[0][1]
        u.pos[2, 0] = u0[0][2]

        u.vel[0, 0] = u0[1][0]
        u.vel[1, 0] = u0[1][1]
        u.vel[2, 0] = u0[1][2]

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
        rhs = acceleration(((3, self.nparts), self.init[1], self.init[2]))
        rhs[:, 0] = part.q[:] / part.m[:] * (f.elec[:, 0] + np.cross(part.vel[:, 0], f.magn[:, 0]))

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
        vel = particles.velocity(((3, 1), self.init[1], self.init[2]))

        Emean = 1.0 / 2.0 * (old_fields.elec + new_fields.elec)

        for n in range(N):
            a = old_parts.q[n] / old_parts.m[n]

            c[:, n] += dt / 2 * a * np.cross(old_parts.vel[:, n], old_fields.magn[:, n] - new_fields.magn[:, n])

            # pre-velocity, separated by the electric forces (and the c term)
            vm = old_parts.vel[:, n] + dt / 2 * a * Emean[:, n] + c[:, n] / 2
            # rotation
            t = dt / 2 * a * new_fields.magn[:, n]
            s = 2 * t / (1 + np.linalg.norm(t, 2) ** 2)
            vp = vm + np.cross(vm + np.cross(vm, t), s)
            # post-velocity
            vel[:, n] = vp + dt / 2 * a * Emean[:, n] + c[:, n] / 2

        return vel
