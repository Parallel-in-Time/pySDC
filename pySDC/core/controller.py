import logging
import os
import sys
from typing import Any, Dict, List, Optional, Type, Union
import numpy as np

from pySDC.core.base_transfer import BaseTransfer
from pySDC.core.errors import ControllerError
from pySDC.helpers.pysdc_helper import FrozenClass
from pySDC.core.check_convergence import CheckConvergence
from pySDC.core.default_hook import DefaultHooks
from pySDC.core.timings import CPUTimings


# short helper class to add params as attributes
class _Pars(FrozenClass):
    def __init__(self, params: Dict[str, Any]) -> None:
        self.mssdc_jac: bool = True
        self.predict_type: Optional[str] = None
        self.all_to_done: bool = False
        self.logger_level: int = 20
        self.log_to_file: bool = False
        self.dump_setup: bool = True
        self.fname: str = 'run_pid' + str(os.getpid()) + '.log'
        self.use_iteration_estimator: bool = False

        for k, v in params.items():
            setattr(self, k, v)

        self._freeze()


class Controller(object):
    """
    Base abstract controller class
    """

    def __init__(
        self, controller_params: Dict[str, Any], description: Dict[str, Any], useMPI: Optional[bool] = None
    ) -> None:
        """
        Initialization routine for the base controller

        Args:
            controller_params (dict): parameter set for the controller and the steps
        """
        self.useMPI: Optional[bool] = useMPI
        self.description: Dict[str, Any] = description

        # check if we have a hook on this list. If not, use default class.
        self.__hooks: List[Any] = []
        hook_classes: List[Type[Any]] = [DefaultHooks, CPUTimings]
        user_hooks = controller_params.get('hook_class', [])
        hook_classes += user_hooks if type(user_hooks) == list else [user_hooks]
        [self.add_hook(hook) for hook in hook_classes]
        controller_params['hook_class'] = hook_classes

        for hook in self.hooks:
            hook.pre_setup(step=None, level_number=None)

        self.params: _Pars = _Pars(controller_params)

        self.__setup_custom_logger(self.params.logger_level, self.params.log_to_file, self.params.fname)
        self.logger: logging.Logger = logging.getLogger('controller')

        if self.params.use_iteration_estimator:
            raise ControllerError(
                'The `use_iteration_estimator` controller parameter has been removed. Its only '
                'implementation lived in `controller_MPI`, was never switched on anywhere, had no '
                'tests, and deadlocked when used: every rank but the last posted a broadcast that '
                'the last rank only matched if the estimate happened to fire. Use the '
                '`CheckIterationEstimatorNonMPI` convergence controller instead, as '
                '`pySDC/tutorial/step_8/C_iteration_estimator.py` does. Note it is, as its name '
                'says, not yet available under MPI.'
            )

        self.base_convergence_controllers: List[Type[Any]] = [CheckConvergence]
        self.setup_convergence_controllers(description)

    @staticmethod
    def __setup_custom_logger(
        level: Optional[int] = None, log_to_file: Optional[bool] = None, fname: Optional[str] = None
    ) -> None:
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

        if level <= logging.DEBUG:
            import warnings

            warnings.warn('Running with debug output will degrade performance as all output is immediately flushed.')

            class StreamFlushingHandler(logging.StreamHandler):
                """
                This will immediately flush any messages to the output.
                """

                def emit(self, record: logging.LogRecord) -> None:
                    super().emit(record)
                    self.flush()

            std_handler = StreamFlushingHandler(sys.stdout)
        else:
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

    def add_hook(self, hook: Type[Any]) -> None:
        """
        Add a hook to the controller which will be called in addition to all other hooks whenever something happens.
        The hook is only added if a hook of the same class is not already present.

        Args:
            hook (pySDC.Hook): A hook class that is derived from the core hook class

        Returns:
            None
        """
        if hook not in [type(me) for me in self.hooks]:
            self.__hooks += [hook()]

    def welcome_message(self) -> None:
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

    def dump_setup(self, step: Any, controller_params: Dict[str, Any], description: Dict[str, Any]) -> None:
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
        for k, v in sorted(vars(self.params).items()):
            if not k.startswith('_'):
                if k in controller_params:
                    out += '--> %s = %s\n' % (k, v)
                else:
                    out += '    %s = %s\n' % (k, v)

        out += '\nStep: %s\n' % step.__class__
        for k, v in sorted(vars(step.params).items()):
            if not k.startswith('_'):
                if k in description['step_params']:
                    out += '--> %s = %s\n' % (k, v)
                else:
                    out += '    %s = %s\n' % (k, v)
        out += f'    Number of steps: {step.status.time_size}\n'

        out += '    Level: %s\n' % step.levels[0].__class__
        for L in step.levels:
            out += '        Level %2i\n' % L.level_index
            for k, v in sorted(vars(L.params).items()):
                if not k.startswith('_'):
                    if k in description['level_params']:
                        out += '-->         %s = %s\n' % (k, v)
                    else:
                        out += '            %s = %s\n' % (k, v)
            out += '-->         Problem: %s\n' % L.prob.__class__
            for k, v in sorted(L.prob.params.items()):
                if k in description['problem_params']:
                    out += '-->             %s = %s\n' % (k, v)
                else:
                    out += '                %s = %s\n' % (k, v)
            out += '-->             Data type u: %s\n' % L.prob.dtype_u
            out += '-->             Data type f: %s\n' % L.prob.dtype_f
            out += '-->             Sweeper: %s\n' % L.sweep.__class__
            for k, v in sorted(vars(L.sweep.params).items()):
                if not k.startswith('_'):
                    if k in description['sweeper_params']:
                        out += '-->                 %s = %s\n' % (k, v)
                    else:
                        out += '                    %s = %s\n' % (k, v)
            out += '-->                 Collocation: %s\n' % L.sweep.coll.__class__

        if len(step.levels) > 1:
            if 'base_transfer_class' in description and description['base_transfer_class'] is not BaseTransfer:
                out += '-->     Base Transfer: %s\n' % step.base_transfer.__class__
            else:
                out += '        Base Transfer: %s\n' % step.base_transfer.__class__
            for k, v in sorted(vars(step.base_transfer.params).items()):
                if not k.startswith('_'):
                    if k in description['base_transfer_params']:
                        out += '-->         %s = %s\n' % (k, v)
                    else:
                        out += '            %s = %s\n' % (k, v)
            out += '-->     Space Transfer: %s\n' % step.base_transfer.space_transfer.__class__
            for k, v in sorted(vars(step.base_transfer.space_transfer.params).items()):
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

    def run(self, u0: Any, t0: float, Tend: float) -> Any:
        """
        Abstract interface to the run() method

        Args:
            u0: initial values
            t0 (float): starting time
            Tend (float): ending time
        """
        raise NotImplementedError('ERROR: controller has to implement run(self, u0, t0, Tend)')

    @property
    def hooks(self) -> List[Any]:
        """
        Getter for the hooks

        Returns:
            pySDC.Hooks.hooks: hooks
        """
        return self.__hooks

    @property
    def steps(self) -> List[Any]:
        """
        Getter for the steps this controller owns.

        Controllers that hold the whole block expose them as `MS`; MPI controllers hold a single step
        as `S`. Dispatch on which of those exists rather than on the class name, so that subclasses
        keep working.

        Returns:
            list: the steps owned by this controller
        """
        return self.MS if hasattr(self, 'MS') else [self.S]

    def check_variable_coefficients(self, num_procs: int) -> None:
        """
        Reject k-dependent QDelta coefficients outside plain SDC.

        MIN-SR-FLEX and the Jumper variants vary QDelta with the sweep index, and the nilpotency
        argument behind them is derived for SDC, where that index *is* the SDC iteration count.
        Anything needing more iterations (PFASST) or fewer (MLSDC) breaks that identity and would
        need its own analysis first.

        They are also only refreshed by `Sweeper.updateVariableCoeffs`, which runs on the finest
        level of the Jacobi sweep alone, so on a coarse level or on the Gauss-Seidel path they
        silently degrade to a fixed preconditioner. Failing loudly beats either of those.

        Note this gates parallelism across *steps* and the number of *levels*. Parallelism across
        collocation nodes (`generic_implicit_MPI` and friends) is still SDC and stays allowed.

        Args:
            num_procs (int): number of parallel time steps

        Raises:
            ControllerError: if a k-dependent QDelta is combined with multiple levels or steps
        """
        S = self.steps[0]
        if len(S.levels) == 1 and num_procs == 1:
            return

        for level in S.levels:
            for name in ['genQI', 'genQE']:
                generator = getattr(level.sweep, name, None)
                if generator is not None and generator.isKDependent():
                    raise ControllerError(
                        f'{type(generator).__name__} varies QDelta with the sweep index and is only '
                        f'verified for SDC, but you have {len(S.levels)} level(s) and {num_procs} '
                        f'step(s). Use a preconditioner with fixed coefficients, e.g. MIN-SR-S or LU.'
                    )

    def setup_convergence_controllers(self, description: Dict[str, Any]) -> None:
        '''
        Setup variables needed for convergence controllers, notably a list containing all of them and a list containing
        their order. Also, we add the `CheckConvergence` convergence controller, which takes care of maximum iteration
        count or a residual based stopping criterion, as well as all convergence controllers added to the description.

        Args:
            description (dict): The description object used to instantiate the controller

        Returns:
            None
        '''
        self.convergence_controllers: List[Any] = []
        # List of indices specifying the order of convergence controllers
        self.convergence_controller_order: List[int] = []
        conv_classes = description.get('convergence_controllers', {})

        # instantiate the convergence controllers
        for conv_class, params in conv_classes.items():
            self.add_convergence_controller(conv_class, description=description, params=params)

        return None

    def add_convergence_controller(
        self,
        convergence_controller: Type[Any],
        description: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
        allow_double: bool = False,
    ) -> None:
        '''
        Add an individual convergence controller to the list of convergence controllers and instantiate it.
        Afterwards, the order of the convergence controllers is updated.

        Args:
            convergence_controller (pySDC.ConvergenceController): The convergence controller to be added
            description (dict): The description object used to instantiate the controller
            params (dict): Parameters for the convergence controller
            allow_double (bool): Allow adding the same convergence controller multiple times

        Returns:
            None
        '''
        # check if we passed any sort of special params
        params = {**({} if params is None else params), 'useMPI': self.useMPI}

        # check if we already have the convergence controller or if we want to have it multiple times
        if convergence_controller not in [type(me) for me in self.convergence_controllers] or allow_double:
            self.convergence_controllers.append(convergence_controller(self, params, description))

            # update ordering
            orders = [C.params.control_order for C in self.convergence_controllers]
            self.convergence_controller_order = np.arange(len(self.convergence_controllers))[np.argsort(orders)]

        return None

    def get_convergence_controllers_as_table(self, description: Dict[str, Any]) -> str:
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

    def return_stats(self) -> Dict[Any, Any]:
        """
        Return the merged stats from all hooks

        Returns:
            dict: Merged stats from all hooks
        """
        stats = {}
        for hook in self.hooks:
            stats = {**stats, **hook.return_stats()}
        return stats


class ParaDiagController(Controller):
    """
    What a ParaDiag controller needs on top of the controller it inherits its driver from.

    This carries no `__init__` of its own. A concrete ParaDiag controller calls
    `prepare_ParaDiag_params` on the two dictionaries and then hands them to the driver's
    initialisation, so that ParaDiag can be mixed into either transport's controller without the
    two having to agree on a constructor signature.
    """

    @staticmethod
    def prepare_ParaDiag_params(controller_params: Dict[str, Any], description: Dict[str, Any]) -> None:
        """
        Check and complete the parameters ParaDiag needs, in place.

        Call this *before* the controller's own initialisation: it only reads and writes the two
        dictionaries, and must have run by the time the steps are built.

        Args:
            controller_params (dict): parameter set for the controller and the steps
            description (dict): all the parameters to set up the rest (levels, problems, ...)
        """
        from pySDC.implementations.sweeper_classes.ParaDiagSweepers import QDiagonalization

        if QDiagonalization in description['sweeper_class'].__mro__:
            description['sweeper_params']['ignore_ic'] = True
            description['sweeper_params']['update_f_evals'] = False
        else:
            logging.getLogger('controller').warning(
                f'Warning: Your sweeper class {description["sweeper_class"]} is not derived from {QDiagonalization}. You probably want to use another sweeper class.'
            )

        if not controller_params.get('all_to_done', True):
            raise NotImplementedError('ParaDiag only implemented with option `all_to_done=True`')
        if 'alpha' not in controller_params.keys():
            from pySDC.core.errors import ParameterError

            raise ParameterError('Please supply alpha as a parameter to the ParaDiag controller!')
        controller_params['average_jacobian'] = controller_params.get('average_jacobian', True)

        controller_params['all_to_done'] = True

    @staticmethod
    def resolve_alpha(alpha: Any, k: int = 0) -> float:
        """
        Read the alpha for iteration `k` out of whatever the user supplied.

        `alpha` may be a single number, a sequence indexed by iteration (the last entry is reused once
        it runs out), or a callable taking the iteration index. Making it iteration dependent lets the
        outer iteration start with a well-conditioned alpha and tighten it later.

        Static because the steps need an alpha before the controller has parameters to read it from.

        Args:
            alpha: the alpha parameter as supplied by the user
            k (int): iteration index

        Returns:
            float: alpha to use for this iteration
        """
        if callable(alpha):
            return float(alpha(k))
        if hasattr(alpha, '__len__'):
            return float(alpha[min(k, len(alpha) - 1)])
        return float(alpha)

    def get_alpha(self, k: int = 0) -> float:
        """
        Get the ParaDiag alpha parameter for iteration `k`.

        Args:
            k (int): iteration index

        Returns:
            float: alpha to use for this iteration
        """
        return self.resolve_alpha(self.params.alpha, k)

    def get_FFT_matrices(self, k: int = 0) -> Any:
        """
        Get the weighted FFT and iFFT matrices for iteration `k`, rebuilding them only when alpha
        actually changes.

        Args:
            k (int): iteration index

        Returns:
            tuple: the forward and backward weighted FFT matrices
        """
        alpha = self.get_alpha(k)
        if getattr(self, '_cached_alpha', None) != alpha:
            from pySDC.helpers.ParaDiagHelper import get_weighted_FFT_matrix, get_weighted_iFFT_matrix

            self._FFT_matrix = get_weighted_FFT_matrix(self.n_steps, alpha)
            self._iFFT_matrix = get_weighted_iFFT_matrix(self.n_steps, alpha)
            self._cached_alpha = alpha
        return self._FFT_matrix, self._iFFT_matrix

    def update_G_inv(self, k: int = 0) -> None:
        """
        Rebuild G^-1 on the local step(s) when alpha changes with the iteration.

        G^-1 depends on alpha, so an iteration dependent alpha means the sweeper's diagonalization has
        to be recomputed. Subclasses implement this because only they know which steps they own.

        Args:
            k (int): iteration index
        """
        raise NotImplementedError('ParaDiag controllers have to implement update_G_inv')

    def FFT_in_time(self, quantity: Any, k: int = 0) -> None:
        """
        Compute weighted forward FFT in time. The weighting is determined by the alpha parameter in ParaDiag

        Note: The implementation via matrix-vector multiplication may be inefficient and less stable compared to an FFT
              with transposes!

        Args:
            quantity (str): the level attribute to transform
            k (int): iteration index, for an iteration dependent alpha
        """
        self.apply_matrix(self.get_FFT_matrices(k)[0], quantity)

    def iFFT_in_time(self, quantity: Any, k: int = 0) -> None:
        """
        Compute weighted backward FFT in time. The weighting is determined by the alpha parameter in ParaDiag

        Args:
            quantity (str): the level attribute to transform
            k (int): iteration index, for an iteration dependent alpha
        """
        self.apply_matrix(self.get_FFT_matrices(k)[1], quantity)
