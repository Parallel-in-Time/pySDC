import numpy as np
from pySDC.core.Errors import ParameterError
from pySDC.implementations.datatype_classes.particles import particles, acceleration
from pySDC.implementations.problem_classes.OuterSolarSystem import outer_solar_system


# noinspection PyUnusedLocal
class full_solar_system(outer_solar_system):
    """
    Example implementing the full solar system problem
    """

    def __init__(self, problem_params, dtype_u=particles, dtype_f=acceleration):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: particle data type (will be passed to parent class)
            dtype_f: acceleration data type (will be passed to parent class)
        """

        if 'sun_only' not in problem_params:
            problem_params['sun_only'] = False

        # these parameters will be used later, so assert their existence
        essential_keys = []
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # invoke parant's class (!) super init, passing nparts, dtype_u and dtype_f
        super(outer_solar_system, self).__init__(((3, 10), None, np.dtype('float64')), dtype_u, dtype_f, problem_params)

        # gravitational constant
        self.G = 2.95912208286E-4

    def u_exact(self, t):
        """
        Routine to compute the exact/initial trajectory at time t

        Args:
            t (float): current time
        Returns:
            dtype_u: exact/initial position and velocity
        """
        assert t == 0.0, 'error, u_exact only works for the initial time t0=0'
        me = self.dtype_u(self.init)

        # initial positions and velocities taken from
        # https://www.aanda.org/articles/aa/full/2002/08/aa1405/aa1405.right.html
        me.pos[:, 0] = [0.0, 0.0, 0.0]
        me.pos[:, 1] = [-2.503321047836E-01, +1.873217481656E-01, +1.260230112145E-01]
        me.pos[:, 2] = [+1.747780055994E-02, -6.624210296743E-01, -2.991203277122E-01]
        me.pos[:, 3] = [-9.091916173950E-01, +3.592925969244E-01, +1.557729610506E-01]
        me.pos[:, 4] = [+1.203018828754E+00, +7.270712989688E-01, +3.009561427569E-01]
        me.pos[:, 5] = [+3.733076999471E+00, +3.052424824299E+00, +1.217426663570E+00]
        me.pos[:, 6] = [+6.164433062913E+00, +6.366775402981E+00, +2.364531109847E+00]
        me.pos[:, 7] = [+1.457964661868E+01, -1.236891078519E+01, -5.623617280033E+00]
        me.pos[:, 8] = [+1.695491139909E+01, -2.288713988623E+01, -9.789921035251E+00]
        me.pos[:, 9] = [-9.707098450131E+00, -2.804098175319E+01, -5.823808919246E+00]

        me.vel[:, 0] = [0.0, 0.0, 0.0]
        me.vel[:, 1] = [-2.438808424736E-02, -1.850224608274E-02, -7.353811537540E-03]
        me.vel[:, 2] = [+2.008547034175E-02, +8.365454832702E-04, -8.947888514893E-04]
        me.vel[:, 3] = [-7.085843239142E-03, -1.455634327653E-02, -6.310912842359E-03]
        me.vel[:, 4] = [-7.124453943885E-03, +1.166307407692E-02, +5.542098698449E-03]
        me.vel[:, 5] = [-5.086540617947E-03, +5.493643783389E-03, +2.478685100749E-03]
        me.vel[:, 6] = [-4.426823593779E-03, +3.394060157503E-03, +1.592261423092E-03]
        me.vel[:, 7] = [+2.647505630327E-03, +2.487457379099E-03, +1.052000252243E-03]
        me.vel[:, 8] = [+2.568651772461E-03, +1.681832388267E-03, +6.245613982833E-04]
        me.vel[:, 9] = [+3.034112963576E-03, -1.111317562971E-03, -1.261841468083E-03]

        # masses relative to the sun taken from
        # https://en.wikipedia.org/wiki/Planetary_mass#Values_from_the_DE405_ephemeris
        me.m[0] = 1.0                   # Sun
        me.m[1] = 0.1660100 * 1E-06     # Mercury
        me.m[2] = 2.4478383 * 1E-06     # Venus
        me.m[3] = 3.0404326 * 1E-06    # Earth+Moon
        me.m[4] = 0.3227151 * 1E-06    # Mars
        me.m[5] = 954.79194 * 1E-06    # Jupiter
        me.m[6] = 285.88600 * 1E-06     # Saturn
        me.m[7] = 43.662440 * 1E-06    # Uranus
        me.m[8] = 51.513890 * 1E-06    # Neptune
        me.m[9] = 0.0073960 * 1E-06     # Pluto

        return me
