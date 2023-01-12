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

        self.base_convergence_controllers = [CheckConvergence]
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
                fmt='%(asctime)s - %(name)s - %(module)s - %(funcName)s - %(lineno)d - %(levelname)s: %(message)s'
            )
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

    def welcome_message(self):
        out = (
            "Welcome to the one and only, really very astonishing and 87.3% bug free"
            + "\n"
            + r"                                 _____ _____   _____ "
            + "\n"
            + r"                                / ____|  __ \ / ____|"
            + "\n"
            + r"                    _ __  _   _| (___ | |  | | |     "
            + "\n"
            + r"                   | '_ \| | | |\___ \| |  | | |     "
            + "\n"
            + r"                   | |_) | |_| |____) | |__| | |____ "
            + "\n"
            + r"                   | .__/ \__, |_____/|_____/ \_____|"
            + "\n"
            + r"                   | |     __/ |                     "
            + "\n"
            + r"                   |_|    |___/                      "
            + "\n"
            + r"                                                     "
        )
        self.logger.info(out)

    def dump_setup(self, step, controller_params, description):
        """
        Helper function to dump the setup used for this controller

        Args:
            step (pySDC.Step.step): the step instance (will/should be the first one only)
            controller_params (dict): controller parameters
            description (dict): description of the problem
        """

        self.welcome_message()
        out = 'Setup overview (--> user-defined, -> dependency) -- BEGIN'
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
            for k, v in L.prob.params.items():
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

        out += '\n'
        out += self.get_convergence_controllers_as_table(description)
        out += '\n'
        self.logger.info(out)

        out = '----------------------------------------------------------------------------------------------------'
        self.logger.info(out)
        out = 'Setup overview (--> user-defined, -> dependency) -- END\n'
        self.logger.info(out)

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
        '''
        Setup variables needed for convergence controllers, notably a list containing all of them and a list containing
        their order. Also, we add the `CheckConvergence` convergence controller, which takes care of maximum iteration
        count or a residual based stopping criterion, as well as all convergence controllers added to the description.

        Args:
            description (dict): The description object used to instantiate the controller

        Returns:
            None
        '''
        self.convergence_controllers = []
        self.convergence_controller_order = []
        conv_classes = description.get('convergence_controllers', {})

        # instantiate the convergence controllers
        for conv_class, params in conv_classes.items():
            self.add_convergence_controller(conv_class, description=description, params=params)

        return None

    def add_convergence_controller(self, convergence_controller, description, params=None, allow_double=False):
        '''
        Add an individual convergence controller to the list of convergence controllers and instantiate it.
        Afterwards, the order of the convergence controllers is updated.

        Args:
            convergence_controller (pySDC.ConvergenceController): The convergence controller to be added
            description (dict): The description object used to instantiate the controller
            params (dict): Parametes for the convergence controller
            allow_double (bool): Allow adding the same convergence controller multiple times

        Returns:
            None
        '''
        # check if we passed any sort of special params
        params = {} if params is None else params

        # check if we already have the convergence controller or if we want to have it multiple times
        if convergence_controller not in [type(me) for me in self.convergence_controllers] or allow_double:
            self.convergence_controllers.append(convergence_controller(self, params, description))

            # update ordering
            orders = [C.params.control_order for C in self.convergence_controllers]
            self.convergence_controller_order = np.arange(len(self.convergence_controllers))[np.argsort(orders)]

        return None

    def get_convergence_controllers_as_table(self, description):
        '''
        This function is for debugging purposes to keep track of the different convergence controllers and their order.

        Args:
            description (dict): Description of the problem

        Returns:
            str: Table of convergence controllers as a string
        '''
        out = 'Active convergence controllers:'
        out += '\n    |  # | order | convergence controller'
        out += '\n----+----+-------+---------------------------------------------------------------------------------------'
        for i in range(len(self.convergence_controllers)):
            C = self.convergence_controllers[self.convergence_controller_order[i]]

            # figure out how the convergence controller was added
            if type(C) in description.get('convergence_controllers', {}).keys():  # added by user
                user_added = '--> '
            elif type(C) in self.base_convergence_controllers:  # added by default
                user_added = '    '
            else:  # added as dependency
                user_added = ' -> '

            out += f'\n{user_added}|{i:3} | {C.params.control_order:5} | {type(C).__name__}'

        return out
