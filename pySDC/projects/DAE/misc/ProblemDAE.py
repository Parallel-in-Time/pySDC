import numpy as np

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh


class ptype_dae(ptype):
    """
    Interface class for DAE problems. Ensures that all parameters are passed that are needed by the DAE sweeper 
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """
        # these parameters will be used by the DAE sweeper, so assert their existence
        essential_keys = ['newton_tol']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)
        # TODO consider defining attributes that save solver run data
        
        super(ptype_dae, self).__init__(
            (problem_params['nvars'], None, np.dtype('float64')), dtype_u, dtype_f, problem_params
        )