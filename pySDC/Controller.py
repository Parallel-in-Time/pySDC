import abc
from future.utils import with_metaclass
import logging

from pySDC.Log import setup_custom_logger
from pySDC.plugins.pysdc_helper import FrozenClass
from pySDC import Hooks as hookclass


class controller(with_metaclass(abc.ABCMeta)):
    """
    Base abstract controller class
    """

    def __init__(self, controller_params):
        """
        Initialization routine for the base controller

        Args:
            controller_params: parameter set for the controller and the steps
        """

        # short helper class to add params as attributes
        class pars(FrozenClass):
            def __init__(self, params):

                self.fine_comm = True
                self.predict = True
                self.logger_level = 20
                self.log_to_file = False

                for k, v in params.items():
                    setattr(self, k, v)

                self._freeze()

        self.params = pars(controller_params)

        setup_custom_logger(self.params.logger_level, self.params.log_to_file)
        self.logger = logging.getLogger('controller')

        # check if we have a hook on this list. if not, use default class.
        if 'hook_class' in controller_params:
            hook = controller_params['hook_class']
        else:
            hook = hookclass.hooks

        self.__hooks = hook()

        pass


    def check_convergence(self, S):
        """
        Routine to determine whether to stop iterating (currently testing the residual and the max. number of iterations)

        Args:
            S: current step

        Returns:
            converged, true or false

        """

        # do all this on the finest level
        L = S.levels[0]

        # get residual and check against prescribed tolerance (plus check number of iterations
        res = L.status.residual
        converged = S.status.iter >= S.params.maxiter or res <= L.params.restol

        return converged

    @abc.abstractmethod
    def run(self, u0, t0, Tend):
        """
        Abstract interface to the run() method

        Args:
            u0: initial values
            t0: starting time
            Tend: ending time
        """
        return None

    @property
    def hooks(self):
        """
        Getter for the hooks
        """
        return self.__hooks
