import logging

from pySDC.helpers.pysdc_helper import FrozenClass


# short helper class to add params as attributes
class _Pars(FrozenClass):
    def __init__(self, pars):

        for k, v in pars.items():
            setattr(self, k, v)

        self._freeze()


class ptype(object):
    """
    Prototype class for problems, just defines the attributes essential to get started

    Attributes:
        logger: custom logger for problem-related logging
        params (__Pars): parameter object containing the custom parameters passed by the user
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
            params (dict): set or parameters
        """

        self.params = _Pars(params)

        # set up logger
        self.logger = logging.getLogger('problem')

        # pass initialization parameter and data types
        self.init = init
        self.dtype_u = dtype_u
        self.dtype_f = dtype_f

    def eval_f(self, u, t):
        """
        Abstract interface to RHS computation of the ODE
        """
        raise NotImplementedError('ERROR: problem has to implement eval_f(self, u, t)')
