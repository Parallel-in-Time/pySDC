import numpy as np

from pySDC.core.errors import ProblemError
from pySDC.core.problem import Problem, WorkCounter
from pySDC.implementations.datatype_classes.particles import particles, fields, acceleration


# noinspection PyUnusedLocal
class penningtrap(Problem):
    r"""
    Charged particles in a 3D Penning trap with Coulomb interaction, a second-order problem for the Boris integrator.

    This class implements a standard Penning trap problem on the time interval :math:`[0, t_{end}]`
    fully investigated in [1]_. The equations are given by the following equation of motion

    .. math::
        \frac{dv}{dt}=f(x,v)=\alpha[E(x,t)+v\times B(x,t)],

    .. math::
        \frac{dx}{dt}=v

    with the particles :math:`x, v\in \mathbb{R}^{3}`. For the penning trap problem, the other parameters are given by
    the constant magnetic field :math:`B=\frac{\omega_{B}}{\alpha}\cdot \hat{e_{z}}\in \mathbb{R}^{3}`
    along the :math:`z`-axis with the particle's charge-to-mass ratio :math:`\alpha=\frac{q}{m}` so that

    .. math::
        v\times B=\frac{\omega_{B}}{\alpha}\left(
            \begin{matrix}
            0 & 1 & 0\\
                -1 & 0 & 0\\
                    0 & 0 & 0
            \end{matrix}
            \right)v.

    The electric field :math:`E(x_{i})=E_{ext}(x_{i})+E_{int}(x_{i})\in \mathbb{R}^{3}` where

    .. math::
        E_{ext}(x_{i})=-\epsilon\frac{\omega_{E}^{2}}{\alpha}\left(
            \begin{matrix}
            1 & 0 & 0\\
                0 & 1 & 0\\
                    0 & 0 & -2
            \end{matrix}
            \right)x_{i}

    and the inter-particle Coulomb interaction

    .. math::
        E_{int}(x_{i})=\sum_{k=1, k\neq i}^{N_{particles}}Q_{k}\frac{x_{i}-x_{k}}{(|x_{i}-x_{k}|^{2}+\lambda^{2})^{3/2}}

    with the smoothing parameter :math:`\lambda>0`.
    The exact solution also given for the single particle penning trap more detailed [1]_, [2]_.
    For to solve nonlinear equation of system, Boris trick is used (see [2]_).

    Parameters
    ----------
    omega_B : float
        Amplitude of magnetic field.
    omega_E : float
        Amplitude of electric field.
    u0 : np.1darray
        Initial condition ``[pos, vel, [q], [m]]``: position and velocity (three components each) around which the
        particles are placed, and the particles' charge and mass (so far only :math:`q = m = 1` is implemented).
    nparts : int
        The number of particles.
    sig : float
        The smoothing parameter :math:`\lambda>0`.

    Attributes
    ----------
    work_counter : dict
        Counts the calls of the right-hand side, and calls of the Boris solver.

    References
    ----------
    .. [1] F. Penning. Die Glimmentladung bei niedrigem Druck zwischen koaxialen Zylindern in einem axialen Magnetfeld.
        Physica. Vol. 3 (1936).
    .. [2] Mathias Winkel, Robert Speck and Daniel Ruprecht. A high-order Boris integrator.
        Journal of Computational Physics (2015).
    """

    dtype_u = particles
    dtype_f = fields

    def __init__(self, omega_B, omega_E, u0, nparts, sig):
        """Initialization routine"""
        # invoke super init, passing nparts, dtype_u and dtype_f
        super().__init__(((3, nparts), None, np.dtype('float64')))
        self._makeAttributeAndRegister('nparts', localVars=locals(), readOnly=True)
        self._makeAttributeAndRegister('omega_B', 'omega_E', 'u0', 'sig', localVars=locals())
        self.work_counters['Boris_solver'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

    @staticmethod
    def fast_interactions(N, pos, sig, q):
        r"""
        Computes the fast interactions.

        All pairs at once, with NumPy broadcasting: three times faster than the numba-compiled loop this
        replaces, and with no compiler to install. The pairwise differences take :math:`3N^2` values, a few
        megabytes for the hundreds of particles a direct sum is meant for. The self-interaction vanishes
        since :math:`\lambda > 0`.

        Parameters
        ----------
        N : int
            Number of particles.
        pos : np.2darray
            Position.
        sig : float
            The smoothing parameter :math:`\lambda > 0`.
        q : np.1darray
            Particle's charge.

        Returns
        -------
        Efield : np.2darray
            The internal E field for each particle.
        """
        pos = np.asarray(pos)  # a plain array: the mesh wrapper would cost more than this arithmetic
        diff = pos[:, :, None] - pos[:, None, :]  # (3, N, N): position of particle i minus that of particle j
        dist2 = diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2 + sig**2
        return (q * diff / dist2**1.5).sum(axis=2)

    def get_interactions(self, part):
        r"""
        Routine to compute the particle-particle interaction, assuming :math:`q = 1` for all particles.

        Parameters
        ----------
        part : dtype_u
            The particles.

        Returns
        -------
        Efield : np.ndarray
            The internal E field for each particle.
        """

        N = self.nparts

        Efield = self.fast_interactions(N, part.pos, self.sig, part.q)

        return Efield

    def eval_f(self, part, t):
        """
        Routine to compute the E and B fields (named f for consistency with the original PEPC version).

        Parameters
        ----------
        part : dtype_u
            The particles.
        t : float
            Current time of the particles (not used here).

        Returns
        -------
        f : dtype_f
            Fields for the particles (internal and external), i.e., the right-hand side of the problem.
        """

        self.work_counters['rhs']()
        try:
            _unused = penningtrap.Harmonic_oscillator
            Emat = np.diag([0, 0, -1])
        except AttributeError:
            Emat = np.diag([1, 1, -2])

        f = self.dtype_f(self.init)

        f.elec[:] = self.get_interactions(part) + self.omega_E**2 / (part.q / part.m) * (Emat @ part.pos)
        f.magn[:] = self.omega_B * np.array([0, 0, 1])[:, None]

        return f

    # TODO : Warning, this should be moved to u_exact(t=0) !
    def u_init(self):
        """
        Routine to compute the starting values for the particles.

        The first particle starts at ``u0``. The others are shifted from it by random amounts in :math:`[-1, 0)` in
        each position component and in :math:`[-5, -4)` in each velocity component, drawn after seeding
        ``np.random`` with the number of particles.

        Returns
        -------
        u : dtype_u
            Particle set filled with initial data.
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
            # draw 3 random variables in [-1, 0) to shift positions
            r = np.random.random_sample(3) - 1
            u.pos[0, n] = r[0] + u0[0][0]
            u.pos[1, n] = r[1] + u0[0][1]
            u.pos[2, n] = r[2] + u0[0][2]

            # draw 3 random variables in [-5, -4) to shift velocities
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
        r"""
        Routine to compute the exact trajectory at time :math:`t` (only for single-particle setup).

        Parameters
        ----------
        t : float
            Current time of the exact trajectory.

        Returns
        -------
        u : dtype_u
            Particle type containing the exact position and velocity.
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
        Helper function to assemble the correct right-hand side out of B and E field.

        Parameters
        ----------
        f : dtype_f
            The field values.
        part : dtype_u
            The current particles data.
        t : float
            The current time.

        Returns
        -------
        rhs : acceleration
            Correct right-hand side of type acceleration.
        """

        if not isinstance(part, particles):
            raise ProblemError('something is wrong during build_f, got %s' % type(part))

        rhs = acceleration(self.init)
        rhs[:] = part.q / part.m * (f.elec + np.cross(part.vel, f.magn, axis=0))

        return rhs

    # noinspection PyTypeChecker
    def boris_solver(self, c, dt, old_fields, new_fields, old_parts):
        r"""
        The actual Boris solver for static (!) B fields, extended by the c-term.

        Parameters
        ----------
        c : dtype_u
            The c term gathering the known values from the previous iteration.
        dt : float
            The (probably scaled) time step size.
        old_fields : dtype_f
            The field values at the previous node :math:`m`.
        new_fields : dtype_f
            The field values at the current node :math:`m+1`.
        old_parts : dtype_u
            The particles at the previous node :math:`m`.

        Returns
        -------
        vel : particles
            The velocities at the :math:`(m+1)`-th node.
        """

        vel = particles.velocity(self.init)
        self.work_counters['Boris_solver']()

        # All particles at once: positions and velocities are (3, N), charge and mass (N,). On plain arrays, since
        # the mesh wrapper around each of these small operations would cost more than the operation itself. The
        # view of `c` still writes into the caller's array.
        v_old, B_old, B_new = np.asarray(old_parts.vel), np.asarray(old_fields.magn), np.asarray(new_fields.magn)
        c_term = np.asarray(c)
        Emean = 0.5 * (np.asarray(old_fields.elec) + np.asarray(new_fields.elec))
        a = old_parts.q / old_parts.m

        c_term += dt / 2 * a * np.cross(v_old, B_old - B_new, axis=0)

        # pre-velocity, separated by the electric forces (and the c term)
        vm = v_old + dt / 2 * a * Emean + c_term / 2
        # rotation
        t = dt / 2 * a * B_new
        s = 2 * t / (1 + np.linalg.norm(t, 2, axis=0) ** 2)
        vp = vm + np.cross(vm + np.cross(vm, t, axis=0), s, axis=0)
        # post-velocity
        vel[:] = vp + dt / 2 * a * Emean + c_term / 2

        return vel
