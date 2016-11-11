import abc
from future.utils import with_metaclass
import logging

from pySDC_core.plugins.pysdc_helper import FrozenClass


class ptype(with_metaclass(abc.ABCMeta)):
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

        # short helper class to add params as attributes
        class __Pars(FrozenClass):
            def __init__(self, pars):

                for k, v in pars.items():
                    setattr(self, k, v)

                self._freeze()

        self.params = __Pars(params)

        # set up logger
        self.logger = logging.getLogger('problem')

        # pass initialization parameter and data types
        self.init = init
        self.dtype_u = dtype_u
        self.dtype_f = dtype_f

    @abc.abstractmethod
    def eval_f(self, u, t):
        """
        Abstract interface to RHS computation of the ODE
        """
        return None
