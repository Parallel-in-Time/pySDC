import numpy as np

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh


class ptype_dae(ptype):
    """
    Interface class for DAE problems. Ensures that all parameters are passed that are needed by DAE sweepers
    """

    def __init__(self, nvars, newton_tol):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """
        super().__init__((nvars, None, np.dtype('float64')), mesh, mesh)
        self._makeAttributeAndRegister('nvars', 'newton_tol', localVars=locals(), readOnly=True)
