import abc
import os
from future.utils import with_metaclass
import logging

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
            controller_params (dict): parameter set for the controller and the steps
        """

        # short helper class to add params as attributes
        class pars(FrozenClass):
            def __init__(self, params):

                self.fine_comm = True
                self.predict = True
                self.logger_level = 20
                self.log_to_file = False
                self.fname = 'run_pid' + str(os.getpid()) + '.log'

                for k, v in params.items():
                    setattr(self, k, v)

                self._freeze()

        self.params = pars(controller_params)

        self.__setup_custom_logger(self.params.logger_level, self.params.log_to_file, self.params.fname)
        self.logger = logging.getLogger('controller')

        # check if we have a hook on this list. if not, use default class.
        if 'hook_class' in controller_params:
            hook = controller_params['hook_class']
        else:
            hook = hookclass.hooks

        self.__hooks = hook()

        pass

    @staticmethod
    def __setup_custom_logger(level=None, log_to_file=None, fname=None):
        """
        Helper function to set main parameters for the logging facility

        Args:
            level (int): level of logging
            log_to_file (bool): flag to turn on/off logging to file
            fname (str):
        """

        assert type(level) is int

        # specify formats and handlers
        if log_to_file:
            file_formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(module)s - %(funcName)s - %(lineno)d - %(levelname)s: %(message)s')
            if os.path.isfile(fname):
                file_handler = logging.FileHandler(fname, mode='a')
            else:
                file_handler = logging.FileHandler(fname, mode='w')
            file_handler.setFormatter(file_formatter)
        else:
            file_handler = None

        std_formatter = logging.Formatter(fmt='%(name)s - %(levelname)s: %(message)s')
        std_handler = logging.StreamHandler()
        std_handler.setFormatter(std_formatter)

        # instantiate logger
        logger = logging.getLogger('')

        # remove handlers from previous calls to controller
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        logger.setLevel(level)
        logger.addHandler(std_handler)
        if log_to_file:
            logger.addHandler(file_handler)
        else:
            pass

    @staticmethod
    def check_convergence(S):
        """
        Routine to determine whether to stop iterating (currently testing the residual + the max. number of iterations)

        Args:
            S (pySDC.Step.step): current step

        Returns:
            bool: converged, true or false

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
            t0 (float): starting time
            Tend (float): ending time
        """
        return None

    @property
    def hooks(self):
        """
        Getter for the hooks

        Returns:
            pySDC.Hooks.hooks: hooks
        """
        return self.__hooks
