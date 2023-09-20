import numpy as np

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.particles import particles, acceleration


# noinspection PyUnusedLocal
class outer_solar_system(ptype):
    r"""
    The :math:`N`-body problem describes the mutual influence of the motion of :math:`N` bodies. Formulation of the problem bases
    on Newton's second law. Therefore, the :math:`N`-body problem is formulated as

    .. math::
        m_i \frac{d^2 {\bf r}_i}{d t^2} = \sum_{j=1, i\neq j}^N G \frac{m_i m_j}{|{\bf r}_i - {\bf r}_j|^3}({\bf r}_i - {\bf r}_j),

    where :math:`m_i` is the :math:`i`-th mass point with position described by the vector :math:`{\bf r}_i`, and :math:`G`
    is the gravitational constant. If only the sun influences the motion of the bodies gravitationally, the equations become

    .. math::
        m_i \frac{d^2 {\bf r}_i}{d t^2} = G \frac{m_1}{|{\bf r}_i - {\bf r}_1|^3}({\bf r}_i - {\bf r}_1).

    This class implements the outer solar system consisting of the six outer planets: the sun, Jupiter, Saturn, Uranus,
    Neptune, and Pluto, i.e., :math:`N=6`.

    Parameters
    ----------
    sun_only : bool, optional
        If False, only the sun is taking into account for the influence of the motion.

    Attributes
    ----------
    G : float
        Gravitational constant.
    """

    dtype_u = particles
    dtype_f = acceleration

    G = 2.95912208286e-4

    def __init__(self, sun_only=False):
        """Initialization routine"""

        # invoke super init, passing nparts
        super().__init__(((3, 6), None, np.dtype('float64')))
        self._makeAttributeAndRegister('sun_only', localVars=locals())

    def eval_f(self, u, t):
        """
        Routine to compute the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            The particles.
        t (float): Current time at which the particles are computed (not used here).

        Returns
        -------
        me : dtype_f
            The right-hand side of the problem.
        """
        me = self.dtype_f(self.init, val=0.0)

        # compute the acceleration due to gravitational forces
        # ... only with respect to the sun
        if self.sun_only:
            for i in range(1, self.init[0][-1]):
                dx = u.pos[:, i] - u.pos[:, 0]
                r = np.sqrt(np.dot(dx, dx))
                df = self.G * dx / (r**3)
                me[:, i] -= u.m[0] * df

        # ... or with all planets involved
        else:
            for i in range(self.init[0][-1]):
                for j in range(i):
                    dx = u.pos[:, i] - u.pos[:, j]
                    r = np.sqrt(np.dot(dx, dx))
                    df = self.G * dx / (r**3)
                    me[:, i] -= u.m[j] * df
                    me[:, j] += u.m[i] * df

        return me

    def u_exact(self, t):
        """
        Routine to compute the exact/initial trajectory at time t.

        Parameters
        ----------
        t : float
            Time of the exact/initial trajectory.

        Returns
        -------
        me : dtype_u
            The exact/initial position and velocity.
        """
        assert t == 0.0, 'error, u_exact only works for the initial time t0=0'
        me = self.dtype_u(self.init)

        me.pos[:, 0] = [0.0, 0.0, 0.0]
        me.pos[:, 1] = [-3.5025653, -3.8169847, -1.5507963]
        me.pos[:, 2] = [9.0755314, -3.0458353, -1.6483708]
        me.pos[:, 3] = [8.3101420, -16.2901086, -7.2521278]
        me.pos[:, 4] = [11.4707666, -25.7294829, -10.8169456]
        me.pos[:, 5] = [-15.5387357, -25.2225594, -3.1902382]

        me.vel[:, 0] = [0.0, 0.0, 0.0]
        me.vel[:, 1] = [0.00565429, -0.00412490, -0.00190589]
        me.vel[:, 2] = [0.00168318, 0.00483525, 0.00192462]
        me.vel[:, 3] = [0.00354178, 0.00137102, 0.00055029]
        me.vel[:, 4] = [0.00288930, 0.00114527, 0.00039677]
        me.vel[:, 5] = [0.00276725, -0.0017072, -0.00136504]

        me.m[0] = 1.00000597682  # Sun
        me.m[1] = 0.000954786104043  # Jupiter
        me.m[2] = 0.000285583733151  # Saturn
        me.m[3] = 0.0000437273164546  # Uranus
        me.m[4] = 0.0000517759138449  # Neptune
        me.m[5] = 1.0 / 130000000.0  # Pluto

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

        ham = 0

        for i in range(self.init[0][-1]):
            ham += 0.5 * u.m[i] * np.dot(u.vel[:, i], u.vel[:, i])

        for i in range(self.init[0][-1]):
            for j in range(i):
                dx = u.pos[:, i] - u.pos[:, j]
                r = np.sqrt(np.dot(dx, dx))
                ham -= self.G * u.m[i] * u.m[j] / r

        return ham
