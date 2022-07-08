import logging
import os
import sys
import numpy as np

from pySDC.core import Hooks as hookclass
from pySDC.core.BaseTransfer import base_transfer
from pySDC.helpers.pysdc_helper import FrozenClass
from pySDC.implementations.convergence_controller_classes.check_convergence import CheckConvergence


# short helper class to add params as attributes
class _Pars(FrozenClass):
    def __init__(self, params):
        self.mssdc_jac = True
        self.predict_type = None
        self.all_to_done = False
        self.logger_level = 20
        self.log_to_file = False
        self.dump_setup = True
        self.fname = 'run_pid' + str(os.getpid()) + '.log'
        self.use_iteration_estimator = False
        self.store_uold = False
        self.use_extrapolation_estimate = False

        for k, v in params.items():
            setattr(self, k, v)

        self._freeze()


class controller(object):
    """
    Base abstract controller class
    """

    def __init__(self, controller_params, description):
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

        if self.params.use_iteration_estimator and self.params.all_to_done:
            self.logger.warning('all_to_done and use_iteration_estimator set, will ignore all_to_done')

        self.setup_convergence_controllers(description)

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

    def setup_convergence_controllers(self, description):
        self.convergence_controllers = []
        self.convergence_controller_order = []
        conv_classes = description.get('convergence_controllers', {})
        conv_classes[CheckConvergence] = {}
        for conv_class, params in conv_classes.items():
            self.add_convergence_controller(conv_class, params, description)

    def add_convergence_controller(self, convergence_controller, params, description, allow_double=False):
        c = convergence_controller(self, params, description)
        if not type(c) in [type(me) for me in self.convergence_controllers] or allow_double:
            self.convergence_controllers.append(c)
            orders = [C.params.control_order for C in self.convergence_controllers]
            self.convergence_controller_order = np.arange(len(self.convergence_controllers))[np.argsort(orders)]

    def convergence_controllers_post_step_processing(self, S):
        # perform the convergence control operations for each controller
        for i in range(len(self.convergence_controllers)):
            self.convergence_controllers[self.convergence_controller_order[i]].post_step_processing(self, S)

    def convergence_controllers_post_iteration_processing(self, S):
        # perform the convergence control operations for each controller
        for i in range(len(self.convergence_controllers)):
            self.convergence_controllers[self.convergence_controller_order[i]].post_iteration_processing(self, S)

    def convergence_control(self, S):
        # perform the convergence control operations for each controller
        for i in range(len(self.convergence_controllers)):
            C = self.convergence_controllers[self.convergence_controller_order[i]]

            C.check_iteration_status(self, S)
            C.get_new_step_size(self, S)
            C.determine_restart(self, S)
