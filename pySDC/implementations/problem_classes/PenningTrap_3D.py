import numpy as np
from numba import jit

from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype, WorkCounter
from pySDC.implementations.datatype_classes.particles import particles, fields, acceleration


# noinspection PyUnusedLocal
class penningtrap(ptype):
    """
    Example implementing particles in a penning trap
    """

    dtype_u = particles
    dtype_f = fields

    def __init__(self, omega_B, omega_E, u0, nparts, sig):
        # invoke super init, passing nparts, dtype_u and dtype_f
        super().__init__(((3, nparts), None, np.dtype('float64')))
        self._makeAttributeAndRegister('nparts', localVars=locals(), readOnly=True)
        self._makeAttributeAndRegister('omega_B', 'omega_E', 'u0', 'sig', localVars=locals())
        self.work_counters['Boris_solver']=WorkCounter()
        self.work_counters['rhs']=WorkCounter()


    @staticmethod
    @jit(nopython=True, nogil=True)
    def fast_interactions(N, pos, sig, q):
        Efield = np.zeros((3, N))
        contrib = np.zeros(3)

        for i in range(N):
            contrib[:] = 0

            for j in range(N):
                dist2 = (
                    (pos[0, i] - pos[0, j]) ** 2
                    + (pos[1, i] - pos[1, j]) ** 2
                    + (pos[2, i] - pos[2, j]) ** 2
                    + sig**2
                )
                contrib += q[j] * (pos[:, i] - pos[:, j]) / dist2**1.5

            Efield[:, i] += contrib[:]

        return Efield

    def get_interactions(self, part):
        """
        Routine to compute the particle-particle interaction, assuming q = 1 for all particles

        Args:
            part (dtype_u): the particles
        Returns:
            numpy.ndarray: the internal E field for each particle

        """

        N = self.nparts

        Efield = self.fast_interactions(N, part.pos, self.sig, part.q)

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

        N = self.nparts

        self.work_counters['rhs']()

        Emat = np.diag([1, 1, -2])
        f = self.dtype_f(self.init)

        f.elec[:] = self.get_interactions(part)

        for n in range(N):
            f.elec[:, n] += self.omega_E**2 / (part.q[n] / part.m[n]) * np.dot(Emat, part.pos[:, n])
            f.magn[:, n] = self.omega_B * np.array([0, 0, 1])

        return f

    # TODO : Warning, this should be moved to u_exact(t=0) !
    def u_init(self):
        """
        Routine to compute the starting values for the particles

        Returns:
            dtype_u: particle set filled with initial data
        """

        u0 = self.u0
        N = self.nparts

        u = self.dtype_u(self.init)

        if u0[2][0] != 1 or u0[3][0] != 1:
            raise ProblemError('so far only q = m = 1 is implemented')

        # set first particle to u0
        u.pos[0, 0] = u0[0][0]
        u.pos[1, 0] = u0[0][1]
        u.pos[2, 0] = u0[0][2]
        u.vel[0, 0] = u0[1][0]
        u.vel[1, 0] = u0[1][1]
        u.vel[2, 0] = u0[1][2]

        u.q[0] = u0[2][0]
        u.m[0] = u0[3][0]

        # initialize random seed
        np.random.seed(N)

        comx = u.pos[0, 0]
        comy = u.pos[1, 0]
        comz = u.pos[2, 0]

        for n in range(1, N):
            # draw 3 random variables in [-1,1] to shift positions
            r = np.random.random_sample(3) - 1
            u.pos[0, n] = r[0] + u0[0][0]
            u.pos[1, n] = r[1] + u0[0][1]
            u.pos[2, n] = r[2] + u0[0][2]

            # draw 3 random variables in [-5,5] to shift velocities
            r = np.random.random_sample(3) - 5
            u.vel[0, n] = r[0] + u0[1][0]
            u.vel[1, n] = r[1] + u0[1][1]
            u.vel[2, n] = r[2] + u0[1][2]

            u.q[n] = u0[2][0]
            u.m[n] = u0[3][0]

            # gather positions to check center
            comx += u.pos[0, n]
            comy += u.pos[1, n]
            comz += u.pos[2, n]

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
        wE = self.omega_E
        wB = self.omega_B
        N = self.nparts
        u0 = self.u0

        if N != 1:
            raise ProblemError('u_exact is only valid for a single particle')

        u = self.dtype_u(((3, 1), self.init[1], self.init[2]))

        wbar = np.sqrt(2) * wE

        # position and velocity in z direction is easy to compute
        u.pos[2, 0] = u0[0][2] * np.cos(wbar * t) + u0[1][2] / wbar * np.sin(wbar * t)
        u.vel[2, 0] = -u0[0][2] * wbar * np.sin(wbar * t) + u0[1][2] * np.cos(wbar * t)

        # define temp. variables to compute complex position
        Op = 1 / 2 * (wB + np.sqrt(wB**2 - 4 * wE**2))
        Om = 1 / 2 * (wB - np.sqrt(wB**2 - 4 * wE**2))
        Rm = (Op * u0[0][0] + u0[1][1]) / (Op - Om)
        Rp = u0[0][0] - Rm
        Im = (Op * u0[0][1] - u0[1][0]) / (Op - Om)
        Ip = u0[0][1] - Im

        # compute position in complex notation
        w = (Rp + Ip * 1j) * np.exp(-Op * t * 1j) + (Rm + Im * 1j) * np.exp(-Om * t * 1j)
        # compute velocity as time derivative of the position
        dw = -1j * Op * (Rp + Ip * 1j) * np.exp(-Op * t * 1j) - 1j * Om * (Rm + Im * 1j) * np.exp(-Om * t * 1j)

        # get the appropriate real and imaginary parts
        u.pos[0, 0] = w.real
        u.vel[0, 0] = dw.real
        u.pos[1, 0] = w.imag
        u.vel[1, 0] = dw.imag

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

        N = self.nparts

        rhs = acceleration(self.init)
        for n in range(N):
            rhs[:, n] = part.q[n] / part.m[n] * (f.elec[:, n] + np.cross(part.vel[:, n], f.magn[:, n]))

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

        N = self.nparts
        vel = particles.velocity(self.init)
        self.work_counters['Boris_solver']()
        Emean = 0.5 * (old_fields.elec + new_fields.elec)
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
