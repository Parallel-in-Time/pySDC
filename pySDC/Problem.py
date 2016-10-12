
from pySDC.plugins.pysdc_helper import FrozenClass

class ptype(object):
    """
    Prototype class for problems, just defines the attributes essential to get started

    Attributes:
        init: number of degrees-of-freedom (whatever this may represent)
        dtype_u: variable data type
        dtype_f: RHS data type
    """

    def __init__(self, init, dtype_u, dtype_f, params):
        """
        Initialization routine

        Args:
            init: number of degrees-of-freedom (whatever this may represent)
            dtype_u: variable data type
            dtype_f: RHS data type
            params: set or parameters
        """

        # short helper class to add params as attributes
        class pars(FrozenClass):
            def __init__(self, params):

                for k, v in params.items():
                    setattr(self, k, v)

                self._freeze()

        self.params = pars(params)

        self.init = init
        self.dtype_u = dtype_u
        self.dtype_f = dtype_f

