import numpy as np

from pySDC.implementations.datatype_classes.particles import particles, acceleration
from pySDC.implementations.problem_classes.OuterSolarSystem import outer_solar_system


# noinspection PyUnusedLocal
class full_solar_system(outer_solar_system):
    r"""
    The :math:`N`-body problem describes the mutual influence of the motion of :math:`N` bodies. Formulation of the problem bases
    on Newton's second law. Therefore, the :math:`N`-body problem is formulated as

    .. math::
        m_i \frac{d^2 {\bf r}_i}{d t^2} = \sum_{j=1, i\neq j}^N G \frac{m_i m_j}{|{\bf r}_i - {\bf r}_j|^3}({\bf r}_i - {\bf r}_j),

    where :math:`m_i` is the :math:`i`-th mass point with position described by the vector :math:`{\bf r}_i`, and :math:`G`
    is the gravitational constant. If only the sun influences the motion of the bodies gravitationally, the equations become

    .. math::
        m_i \frac{d^2 {\bf r}_i}{d t^2} = G \frac{m_1}{|{\bf r}_i - {\bf r}_1|^3}({\bf r}_i - {\bf r}_1).

    This class implements the full solar system containing all planets including earth's moon, i.e., :math:`N=10`. Initial conditions
    are taken from [1]_, and masses relative to the sun taken from [2]_.

    Parameters
    ----------
    sun_only : bool, optional
        If False, only the sun is taken into account for the influence of the motion.

    Attributes
    ----------
    G : float
        Gravitational constant.

    References
    ----------
    .. [1] https://www.aanda.org/articles/aa/full/2002/08/aa1405/aa1405.right.html
    .. [2] https://en.wikipedia.org/wiki/Planetary_mass#Values_from_the_DE405_ephemeris
    """

    dtype_u = particles
    dtype_f = acceleration

    def __init__(self, sun_only=False):
        """Initialization routine"""

        super().__init__(sun_only)
        self.init = ((3, 10), None, np.dtype('float64'))

    def u_exact(self, t):
        """
        Routine to compute the exact/initial trajectory at time t. Values here are taken from [1]_, [2]_.

        Parameters
        ----------
        t : float
            Time of the exact/initial trajectory.

        Returns
        -------
        me : dtype_u
            Exact/initial position and velocity.
        """
        assert t == 0.0, 'error, u_exact only works for the initial time t0=0'
        me = self.dtype_u(self.init)

        me.pos[:, 0] = [0.0, 0.0, 0.0]
        me.pos[:, 1] = [-2.503321047836e-01, +1.873217481656e-01, +1.260230112145e-01]
        me.pos[:, 2] = [+1.747780055994e-02, -6.624210296743e-01, -2.991203277122e-01]
        me.pos[:, 3] = [-9.091916173950e-01, +3.592925969244e-01, +1.557729610506e-01]
        me.pos[:, 4] = [+1.203018828754e00, +7.270712989688e-01, +3.009561427569e-01]
        me.pos[:, 5] = [+3.733076999471e00, +3.052424824299e00, +1.217426663570e00]
        me.pos[:, 6] = [+6.164433062913e00, +6.366775402981e00, +2.364531109847e00]
        me.pos[:, 7] = [+1.457964661868e01, -1.236891078519e01, -5.623617280033e00]
        me.pos[:, 8] = [+1.695491139909e01, -2.288713988623e01, -9.789921035251e00]
        me.pos[:, 9] = [-9.707098450131e00, -2.804098175319e01, -5.823808919246e00]

        me.vel[:, 0] = [0.0, 0.0, 0.0]
        me.vel[:, 1] = [-2.438808424736e-02, -1.850224608274e-02, -7.353811537540e-03]
        me.vel[:, 2] = [+2.008547034175e-02, +8.365454832702e-04, -8.947888514893e-04]
        me.vel[:, 3] = [-7.085843239142e-03, -1.455634327653e-02, -6.310912842359e-03]
        me.vel[:, 4] = [-7.124453943885e-03, +1.166307407692e-02, +5.542098698449e-03]
        me.vel[:, 5] = [-5.086540617947e-03, +5.493643783389e-03, +2.478685100749e-03]
        me.vel[:, 6] = [-4.426823593779e-03, +3.394060157503e-03, +1.592261423092e-03]
        me.vel[:, 7] = [+2.647505630327e-03, +2.487457379099e-03, +1.052000252243e-03]
        me.vel[:, 8] = [+2.568651772461e-03, +1.681832388267e-03, +6.245613982833e-04]
        me.vel[:, 9] = [+3.034112963576e-03, -1.111317562971e-03, -1.261841468083e-03]

        me.m[0] = 1.0  # Sun
        me.m[1] = 0.1660100 * 1e-06  # Mercury
        me.m[2] = 2.4478383 * 1e-06  # Venus
        me.m[3] = 3.0404326 * 1e-06  # Earth+Moon
        me.m[4] = 0.3227151 * 1e-06  # Mars
        me.m[5] = 954.79194 * 1e-06  # Jupiter
        me.m[6] = 285.88600 * 1e-06  # Saturn
        me.m[7] = 43.662440 * 1e-06  # Uranus
        me.m[8] = 51.513890 * 1e-06  # Neptune
        me.m[9] = 0.0073960 * 1e-06  # Pluto

        return me
