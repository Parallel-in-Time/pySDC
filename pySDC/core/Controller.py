import logging
import os
import sys

from pySDC.core import Hooks as hookclass
from pySDC.core.BaseTransfer import base_transfer
from pySDC.helpers.pysdc_helper import FrozenClass


# short helper class to add params as attributes
class _Pars(FrozenClass):
    def __init__(self, params):
        self.fine_comm = True
        self.mssdc_jac = True
        self.predict_type = None
        self.all_to_done = False
        self.logger_level = 20
        self.log_to_file = False
        self.dump_setup = True
        self.fname = 'run_pid' + str(os.getpid()) + '.log'
        self.use_iteration_estimator = False

        for k, v in params.items():
            setattr(self, k, v)

        self._freeze()


class controller(object):
    """
    Base abstract controller class
    """

    def __init__(self, controller_params):
        """
        Initialization routine for the base controller

        Args:
            controller_params (dict): parameter set for the controller and the steps
        """

        # check if we have a hook on this list. if not, use default class.
        controller_params['hook_class'] = controller_params.get('hook_class', hookclass.hooks)
        self.__hooks = controller_params['hook_class']()

        self.hooks.pre_setup(step=None, level_number=None)

        self.params = _Pars(controller_params)

        self.__setup_custom_logger(self.params.logger_level, self.params.log_to_file, self.params.fname)
        self.logger = logging.getLogger('controller')

        # if self.params.dump_setup and self.params.logger_level > 20:
        #     self.logger.warning('Will not dump setup, logging level is too high, need at most 20')

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

    def dump_setup(self, step, controller_params, description):
        """
        Helper function to dump the setup used for this controller

        Args:
            step (pySDC.Step.step): the step instance (will/should be the first one only)
            controller_params (dict): controller parameters
            description (dict): description of the problem
        """

        out = 'Setup overview (--> user-defined) -- BEGIN'
        self.logger.info(out)
        out = '----------------------------------------------------------------------------------------------------\n\n'
        out += 'Controller: %s\n' % self.__class__
        for k, v in vars(self.params).items():
            if not k.startswith('_'):
                if k in controller_params:
                    out += '--> %s = %s\n' % (k, v)
                else:
                    out += '    %s = %s\n' % (k, v)

        out += '\nStep: %s\n' % step.__class__
        for k, v in vars(step.params).items():
            if not k.startswith('_'):
                if k in description['step_params']:
                    out += '--> %s = %s\n' % (k, v)
                else:
                    out += '    %s = %s\n' % (k, v)

        out += '    Level: %s\n' % step.levels[0].__class__
        for L in step.levels:
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

        if len(step.levels) > 1:
            if 'base_transfer_class' in description and description['base_transfer_class'] is not base_transfer:
                out += '-->     Base Transfer: %s\n' % step.base_transfer.__class__
            else:
                out += '        Base Transfer: %s\n' % step.base_transfer.__class__
            for k, v in vars(step.base_transfer.params).items():
                if not k.startswith('_'):
                    if k in description['base_transfer_params']:
                        out += '-->         %s = %s\n' % (k, v)
                    else:
                        out += '            %s = %s\n' % (k, v)
            out += '-->     Space Transfer: %s\n' % step.base_transfer.space_transfer.__class__
            for k, v in vars(step.base_transfer.space_transfer.params).items():
                if not k.startswith('_'):
                    if k in description['space_transfer_params']:
                        out += '-->         %s = %s\n' % (k, v)
                    else:
                        out += '            %s = %s\n' % (k, v)
        self.logger.info(out)
        out = '----------------------------------------------------------------------------------------------------'
        self.logger.info(out)
        out = 'Setup overview (--> user-defined) -- END\n'
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
        converged = S.status.iter >= S.params.maxiter or res <= L.params.restol or S.status.force_done
        if converged is None:
            converged = False

        return converged

    def run(self, u0, t0, Tend):
        """
        Abstract interface to the run() method

        Args:
            u0: initial values
            t0 (float): starting time
            Tend (float): ending time
        """
        raise NotImplementedError('ERROR: controller has to implement run(self, u0, t0, Tend)')

    @property
    def hooks(self):
        """
        Getter for the hooks

        Returns:
            pySDC.Hooks.hooks: hooks
        """
        return self.__hooks
