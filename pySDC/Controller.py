import abc
import os
import sys
from future.utils import with_metaclass
import logging
import coloredlogs

from pySDC.plugins.pysdc_helper import FrozenClass
from pySDC import Hooks as hookclass
from pySDC.BaseTransfer import base_transfer


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
                self.log_to_file = True
                self.dump_setup = True
                self.fname = 'run_pid' + str(os.getpid()) + '.log'

                for k, v in params.items():
                    setattr(self, k, v)

                self._freeze()

        self.params = pars(controller_params)

        self.__setup_custom_logger(self.params.logger_level, self.params.log_to_file, self.params.fname)
        self.logger = logging.getLogger('controller')

        if self.params.dump_setup and self.params.logger_level > 20:
            self.logger.warning('Will not dump setup, logging level is too high, need at most 20')

        # check if we have a hook on this list. if not, use default class.
        controller_params['hook_class'] = controller_params.get('hook_class', hookclass.hooks)

        self.__hooks = controller_params['hook_class']()

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

        std_formatter = coloredlogs.ColoredFormatter(fmt='%(name)s - %(levelname)s: %(message)s')
        std_handler = logging.StreamHandler(sys.stdout)
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

    def dump_setup(self, controller_params, description):
        """
        Helper function to dump the setup used for this controller
        """

        out = 'Setup overview (user-defined)\n\n'
        out += 'Controller: %s\n' % self.__class__
        for k, v in vars(self.params).items():
            if not k.startswith('_'):
                if k in controller_params:
                    out += '--> %s = %s\n' % (k, v)
                else:
                    out += '    %s = %s\n' % (k, v)

        out += '\nStep: %s\n' % self.MS[0].__class__
        for k, v in vars(self.MS[0].params).items():
            if not k.startswith('_'):
                if k in description['step_params']:
                    out += '--> %s = %s\n' % (k, v)
                else:
                    out += '    %s = %s\n' % (k, v)

        out += '    Level: %s\n' % self.MS[0].levels[0].__class__
        for L in self.MS[0].levels:
            out += '        Level %2i\n' % L.level_index
            for k, v in vars(L.params).items():
                if not k.startswith('_'):
                    if k in description['level_params']:
                        out += '-->         %s = %s\n' % (k, v)
                    else:
                        out += '            %s = %s\n' % (k, v)
            out += '-->         Problem: %s\n' % L.prob.__class__
            for k, v in vars(L.prob.params).items():
                if not k.startswith('_'):
                    if k in description['problem_params']:
                        out += '-->             %s = %s\n' % (k, v)
                    else:
                        out += '                %s = %s\n' % (k, v)
            out += '-->             Data type u: %s\n' % L.prob.dtype_u
            out += '-->             Data type f: %s\n' % L.prob.dtype_f
            out += '-->             Sweeper: %s\n' % L.sweep.__class__
            for k, v in vars(L.sweep.params).items():
                if not k.startswith('_'):
                    if k in description['sweeper_params']:
                        out += '-->                 %s = %s\n' % (k, v)
                    else:
                        out += '                    %s = %s\n' % (k, v)
            out += '-->                 Collocation: %s\n' % L.sweep.coll.__class__

        if len(self.MS[0].levels) > 1:
            if description['base_transfer_class'] is not base_transfer:
                out += '-->     Base Transfer: %s\n' % self.MS[0].base_transfer.__class__
            else:
                out += '        Base Transfer: %s\n' % self.MS[0].base_transfer.__class__
            for k, v in vars(self.MS[0].base_transfer.params).items():
                if not k.startswith('_'):
                    if k in description['base_transfer_params']:
                        out += '-->         %s = %s\n' % (k, v)
                    else:
                        out += '            %s = %s\n' % (k, v)
            out += '-->     Space Transfer: %s\n' % self.MS[0].base_transfer.space_transfer.__class__
            for k, v in vars(self.MS[0].base_transfer.space_transfer.params).items():
                if not k.startswith('_'):
                    if k in description['space_transfer_params']:
                        out += '-->         %s = %s\n' % (k, v)
                    else:
                        out += '            %s = %s\n' % (k, v)
        out += '\n'
        self.logger.info(out)

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
