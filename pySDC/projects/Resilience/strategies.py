import numpy as np
from matplotlib.colors import TABLEAU_COLORS

cmap = TABLEAU_COLORS


def merge_descriptions(descA, descB):
    """
    Merge two dictionaries that may contain dictionaries, which happens when merging descriptions, for instance.

    Keys that occur in both dictionaries will be overwritten by the ones from `descB` and `descA` will be modified, not
    copied!

    Args:
        descA (dict): Dictionary that you want to merge into
        descB (dict): Dictionary you want to merge from

    Returns:
        dict: decsA with updated parameters
    """
    for key in descB.keys():
        if type(descB[key]) == dict:
            descA[key] = merge_descriptions(descA.get(key, {}), descB[key])
        else:
            descA[key] = descB[key]
    return descA


class Strategy:
    '''
    Abstract class for resilience strategies
    '''

    def __init__(self, useMPI=False, skip_residual_computation='none'):
        '''
        Initialization routine
        '''
        self.useMPI = useMPI
        self.max_steps = 1e4

        # set default values for plotting
        self.linestyle = '-'
        self.marker = '.'
        self.name = ''
        self.bar_plot_x_label = ''
        self.color = list(cmap.values())[0]

        # parameters for computational efficiency
        if skip_residual_computation == 'all':
            self.skip_residual_computation = ('IT_CHECK', 'IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE')
        elif skip_residual_computation == 'most':
            self.skip_residual_computation = ('IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE')
        else:
            self.skip_residual_computation = ()

        # setup custom descriptions
        self.custom_description = {}
        self.custom_description['sweeper_params'] = {'skip_residual_computation': self.skip_residual_computation}

        # prepare parameters for masks to identify faults that cannot be fixed by this strategy
        self.fixable = []
        self.fixable += [
            {
                'key': 'node',
                'op': 'gt',
                'val': 0,
            }
        ]
        self.fixable += [
            {
                'key': 'error',
                'op': 'isfinite',
            }
        ]

        # stuff for work-precision diagrams
        self.precision_parameter = None
        self.precision_parameter_loc = []

    def __str__(self):
        return self.name

    def get_fixable_params(self, **kwargs):
        """
        Return a list containing dictionaries which can be passed to `FaultStats.get_mask` as keyword arguments to
        obtain a mask of faults that can be fixed

        Returns:
            list: Dictionary of parameters
        """
        return self.fixable

    def get_fault_args(self, problem, num_procs):
        '''
        Routine to get arguments for the faults that are exempt from randomization

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            dict: Arguments for the faults that are exempt from randomization
        '''
        args = {}
        args['target'] = 0

        if problem.__name__ == "run_vdp":
            args['time'] = 5.25
        elif problem.__name__ == "run_Schroedinger":
            args['time'] = 0.3
        elif problem.__name__ == "run_quench":
            args['time'] = 31.0
        elif problem.__name__ == "run_Lorenz":
            args['time'] = 0.3

        return args

    def get_random_params(self, problem, num_procs):
        '''
        Routine to get parameters for the randomization of faults

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            dict: Randomization parameters
        '''
        base_params = self.get_base_parameters(problem, num_procs)

        rnd_params = {}
        rnd_params['iteration'] = base_params['step_params']['maxiter']
        rnd_params['rank'] = num_procs

        if problem.__name__ in ['run_Schroedinger', 'run_quench']:
            rnd_params['min_node'] = 1

        if problem.__name__ == "run_quench":
            rnd_params['iteration'] = 1
        return rnd_params

    @property
    def style(self):
        """
        Get the plotting parameters for the strategy.
        Supply them to a plotting function using `**`

        Returns:
            (dict): The plotting parameters as a dictionary
        """
        return {
            'marker': self.marker,
            'label': self.label,
            'color': self.color,
            'ls': self.linestyle,
        }

    @property
    def label(self):
        """
        Get a label for plotting
        """
        return self.name

    @classmethod
    def get_Tend(cls, problem, num_procs=1):
        '''
        Get the final time of runs for fault stats based on the problem

        Args:
            problem (function): A problem to run
            num_procs (int): Number of processes

        Returns:
            float: Tend to put into the run
        '''
        if problem.__name__ == "run_vdp":
            return 11.5
            # return 2.3752559741400825  # old stuff
        elif problem.__name__ == "run_piline":
            return 20.0
        elif problem.__name__ == "run_Lorenz":
            return 1.5
        elif problem.__name__ == "run_Schroedinger":
            return 1.0
        elif problem.__name__ == "run_quench":
            return 500.0
        else:
            raise NotImplementedError('I don\'t have a final time for your problem!')

    def get_base_parameters(self, problem, num_procs=1):
        '''
        Get a base parameters for the problems independent of the strategy.

        Args:
            problem (function): A problem to run
            num_procs (int): Number of processes

        Returns:
            dict: Custom description
        '''
        from pySDC.implementations.convergence_controller_classes.step_size_limiter import StepSizeLimiter

        custom_description = {}
        if problem.__name__ == "run_vdp":
            custom_description['step_params'] = {'maxiter': 3}
            custom_description['problem_params'] = {
                'u0': np.array([2, 0], dtype=np.float64),
                # 'u0': np.array([0.99995, -0.00999985], dtype=np.float64),  # old stuff
                'crash_at_maxiter': False,
                'newton_tol': 1e-11,
            }
            custom_description['level_params'] = {'dt': 1e-2}

        elif problem.__name__ == "run_Lorenz":
            custom_description['step_params'] = {'maxiter': 5}
            custom_description['level_params'] = {'dt': 1e-2}
        elif problem.__name__ == "run_Schroedinger":
            custom_description['step_params'] = {'maxiter': 5}
            custom_description['level_params'] = {'dt': 1e-2, 'restol': -1}
        elif problem.__name__ == "run_quench":
            custom_description['level_params'] = {'restol': -1, 'dt': 8.0}
            custom_description['step_params'] = {'maxiter': 5}
            custom_description['problem_params'] = {'newton_iter': 99, 'newton_tol': 1e-11}

        custom_description['convergence_controllers'] = {
            StepSizeLimiter: {'dt_min': self.get_Tend(problem=problem, num_procs=num_procs) / self.max_steps}
        }
        return custom_description

    def get_custom_description(self, problem, num_procs=1):
        '''
        Get a custom description based on the problem

        Args:
            problem (function): A problem to run
            num_procs (int): Number of processes

        Returns:
            dict: Custom description
        '''
        custom_description = self.get_base_parameters(problem, num_procs)
        return merge_descriptions(custom_description, self.custom_description)

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        raise NotImplementedError('The reference value you are looking for is not implemented for this strategy!')


class BaseStrategy(Strategy):
    '''
    Do a fixed iteration count
    '''

    def __init__(self, useMPI=False, skip_residual_computation='all'):
        '''
        Initialization routine
        '''
        super().__init__(useMPI=useMPI, skip_residual_computation=skip_residual_computation)
        self.color = list(cmap.values())[0]
        self.marker = 'o'
        self.name = 'base'
        self.bar_plot_x_label = 'base'
        self.precision_parameter = 'dt'
        self.precision_parameter_loc = ['level_params', 'dt']

    @property
    def label(self):
        return r'fixed'

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_vdp":
            if key == 'work_newton' and op == sum:
                return 12453
            elif key == 'e_global_post_run' and op == max:
                return 4.3956128381594795e-06

        raise NotImplementedError('The reference value you are looking for is not implemented for this strategy!')


class AdaptivityStrategy(Strategy):
    '''
    Adaptivity as a resilience strategy
    '''

    def __init__(self, useMPI=False, skip_residual_computation='all'):
        '''
        Initialization routine
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

        super().__init__(useMPI=useMPI, skip_residual_computation=skip_residual_computation)
        self.color = list(cmap.values())[1]
        self.marker = '*'
        self.name = 'adaptivity'
        self.bar_plot_x_label = 'adaptivity'
        self.precision_parameter = 'e_tol'
        self.precision_parameter_loc = ['convergence_controllers', Adaptivity, 'e_tol']

    @property
    def label(self):
        return r'$\Delta t$ adaptivity'

    def get_fixable_params(self, maxiter, **kwargs):
        """
        Here faults occurring in the last iteration cannot be fixed.

        Args:
            maxiter (int): Max. iterations until convergence is declared

        Returns:
            (list): Contains dictionaries of keyword arguments for `FaultStats.get_mask`
        """
        self.fixable += [
            {
                'key': 'iteration',
                'op': 'lt',
                'val': maxiter,
            }
        ]
        return self.fixable

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds adaptivity

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom descriptions you can supply to the problem when running it
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

        custom_description = {}
        custom_description['convergence_controllers'] = {}

        dt_max = np.inf
        dt_min = 1e-5
        dt_slope_max = np.inf

        if problem.__name__ == "run_piline":
            e_tol = 1e-7
            dt_min = 1e-2
        elif problem.__name__ == "run_vdp":
            e_tol = 2e-5
            dt_min = 1e-3
        elif problem.__name__ == "run_Lorenz":
            e_tol = 2e-5
            dt_min = 1e-3
        elif problem.__name__ == "run_Schroedinger":
            e_tol = 4e-6
            dt_min = 1e-3
        elif problem.__name__ == "run_quench":
            e_tol = 1e-5
            dt_min = 1e-3
            # dt_max = 25
            # dt_slope_max = 4.

            from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestarting

            custom_description['convergence_controllers'][BasicRestarting.get_implementation(useMPI=self.useMPI)] = {
                'max_restarts': 15,
            }
        else:
            raise NotImplementedError(
                'I don\'t have a tolerance for adaptivity for your problem. Please add one to the\
 strategy'
            )

        custom_description['convergence_controllers'][Adaptivity] = {
            'e_tol': e_tol,
            'dt_min': dt_min,
            'dt_max': dt_max,
            'dt_slope_max': dt_slope_max,
        }
        return merge_descriptions(super().get_custom_description(problem, num_procs), custom_description)

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_vdp":
            if key == 'work_newton' and op == sum:
                return 3825
            elif key == 'e_global_post_run' and op == max:
                return 1.3370376368393444e-05

        raise NotImplementedError('The reference value you are looking for is not implemented for this strategy!')


class AdaptivityRestartFirstStep(AdaptivityStrategy):
    def __init__(self, useMPI=False):
        super().__init__(useMPI=useMPI)
        self.color = 'teal'
        self.name = 'adaptivityRestartFirstStep'

    def get_custom_description(self, problem, num_procs):
        '''
        Add the other version of basic restarting.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom descriptions you can supply to the problem when running it
        '''
        custom_description = super().get_custom_description(problem, num_procs)
        from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestarting

        custom_description['convergence_controllers'][BasicRestarting.get_implementation(useMPI=self.useMPI)] = {
            'max_restarts': 15,
            'restart_from_first_step': True,
        }
        return custom_description

    @property
    def label(self):
        return f'{super().label} restart from first step'


class AdaptiveHotRodStrategy(Strategy):
    '''
    Adaptivity + Hot Rod as a resilience strategy
    '''

    def __init__(self, useMPI=False, skip_residual_computation='all'):
        '''
        Initialization routine
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

        super().__init__(useMPI=useMPI, skip_residual_computation=skip_residual_computation)
        self.color = list(cmap.values())[4]
        self.marker = '.'
        self.name = 'adaptive Hot Rod'
        self.bar_plot_x_label = 'adaptive\nHot Rod'
        self.precision_parameter = 'e_tol'
        self.precision_parameter_loc = ['convergence_controllers', Adaptivity, 'e_tol']

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds adaptivity and Hot Rod

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom description you can supply to the problem when running it
        '''
        from pySDC.implementations.convergence_controller_classes.hotrod import HotRod
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

        if problem.__name__ == "run_vdp":
            e_tol = 3e-7
            dt_min = 1e-3
            maxiter = 4
            HotRod_tol = 2e-6
        else:
            raise NotImplementedError(
                'I don\'t have a tolerance for adaptive Hot Rod for your problem. Please add one \
to the strategy'
            )

        no_storage = num_procs > 1

        custom_description = {
            'convergence_controllers': {
                HotRod: {'HotRod_tol': HotRod_tol, 'no_storage': no_storage},
                Adaptivity: {'e_tol': e_tol, 'dt_min': dt_min, 'embedded_error_flavor': 'linearized'},
            },
            'step_params': {'maxiter': maxiter},
        }

        return merge_descriptions(super().get_custom_description(problem, num_procs), custom_description)

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_vdp":
            if key == 'work_newton' and op == sum:
                return 4466
            elif key == 'e_global_post_run' and op == max:
                return 2.1455229857747504e-06

        raise NotImplementedError('The reference value you are looking for is not implemented for this strategy!')


class IterateStrategy(Strategy):
    '''
    Iterate for as much as you want
    '''

    def __init__(self, useMPI=False, skip_residual_computation='most'):
        '''
        Initialization routine
        '''
        super().__init__(useMPI=useMPI, skip_residual_computation=skip_residual_computation)
        self.color = list(cmap.values())[2]
        self.marker = 'v'
        self.name = 'iterate'
        self.bar_plot_x_label = 'iterate'
        self.precision_parameter = 'restol'
        self.precision_parameter_loc = ['level_params', 'restol']

    @property
    def label(self):
        return r'$k$ adaptivity'

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that allows for adaptive iteration counts

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom description you can supply to the problem when running it
        '''
        restol = -1
        e_tol = -1

        if problem.__name__ == "run_piline":
            restol = 2.3e-8
        elif problem.__name__ == "run_vdp":
            restol = 9e-7
        elif problem.__name__ == "run_Lorenz":
            restol = 16e-7
        elif problem.__name__ == "run_Schroedinger":
            restol = 6.5e-7
        elif problem.__name__ == "run_quench":
            restol = 1e-7
        else:
            raise NotImplementedError(
                'I don\'t have a residual tolerance for your problem. Please add one to the \
strategy'
            )

        custom_description = {
            'step_params': {'maxiter': 99},
            'level_params': {'restol': restol, 'e_tol': e_tol},
        }

        if problem.__name__ == "run_quench":
            custom_description['level_params']['dt'] = 1

        return merge_descriptions(super().get_custom_description(problem, num_procs), custom_description)

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_vdp":
            if key == 'work_newton' and op == sum:
                return 8534
            elif key == 'e_global_post_run' and op == max:
                return 0.0005961192269257065

        raise NotImplementedError('The reference value you are looking for is not implemented for this strategy!')


class HotRodStrategy(Strategy):
    '''
    Hot Rod as a resilience strategy
    '''

    def __init__(self, useMPI=False, skip_residual_computation='all'):
        '''
        Initialization routine
        '''
        super().__init__(useMPI=useMPI, skip_residual_computation=skip_residual_computation)
        self.color = list(cmap.values())[3]
        self.marker = '^'
        self.name = 'Hot Rod'
        self.bar_plot_x_label = 'Hot Rod'
        self.precision_parameter = 'dt'
        self.precision_parameter_loc = ['level_params', 'dt']

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds Hot Rod

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom description you can supply to the problem when running it
        '''
        from pySDC.implementations.convergence_controller_classes.hotrod import HotRod
        from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestartingNonMPI

        if problem.__name__ == "run_vdp":
            if num_procs == 4:
                HotRod_tol = 1.800804e-04
            elif num_procs == 5:
                HotRod_tol = 9.329361e-05
            else:  # 1 process
                HotRod_tol = 1.347949e-06  # 5e-7
            HotRod_tol = 7e-6 if num_procs > 1 else 5e-7
            maxiter = 4
        elif problem.__name__ == "run_Lorenz":
            if num_procs == 5:
                HotRod_tol = 9.539348e-06
            elif num_procs == 4:
                HotRod_tol = 3.201e-6
            else:
                HotRod_tol = 7.720589e-07  # 4e-7
            maxiter = 6
        elif problem.__name__ == "run_Schroedinger":
            if num_procs == 5:
                HotRod_tol = 2.497697e-06
            elif num_procs == 4:
                HotRod_tol = 1.910405e-06
            else:
                HotRod_tol = 4.476790e-07
            maxiter = 6
        elif problem.__name__ == "run_quench":
            if num_procs == 5:
                HotRod_tol = 1.017534e-03
            elif num_procs == 4:
                HotRod_tol = 1.017534e-03
            else:
                HotRod_tol = 5.198620e-04
            maxiter = 6
        else:
            raise NotImplementedError(
                'I don\'t have a tolerance for Hot Rod for your problem. Please add one to the\
 strategy'
            )

        no_storage = False  # num_procs > 1

        custom_description = {
            'convergence_controllers': {
                HotRod: {'HotRod_tol': HotRod_tol, 'no_storage': no_storage},
                BasicRestartingNonMPI: {
                    'max_restarts': 2,
                    'crash_after_max_restarts': False,
                    'restart_from_first_step': True,
                },
            },
            'step_params': {'maxiter': maxiter},
        }

        return merge_descriptions(super().get_custom_description(problem, num_procs), custom_description)

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_vdp":
            if key == 'work_newton' and op == sum:
                return 15230
            elif key == 'e_global_post_run' and op == max:
                return 4.3956128381594795e-06

        raise NotImplementedError('The reference value you are looking for is not implemented for this strategy!')


class AdaptivityCollocationStrategy(Strategy):
    '''
    Adaptivity based on collocation as a resilience strategy
    '''

    def __init__(self, useMPI=False, skip_residual_computation='most'):
        '''
        Initialization routine
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityCollocation

        super().__init__(useMPI=useMPI, skip_residual_computation=skip_residual_computation)
        self.color = list(cmap.values())[1]
        self.marker = '*'
        self.name = 'adaptivity_coll'
        self.bar_plot_x_label = 'adaptivity collocation'
        self.precision_parameter = 'e_tol'
        self.adaptive_coll_params = {}
        self.precision_parameter_loc = ['convergence_controllers', AdaptivityCollocation, 'e_tol']
        self.restol = None
        self.maxiter = 99

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds adaptivity

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom descriptions you can supply to the problem when running it
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityCollocation

        custom_description = {}
        custom_description['step_params'] = {'maxiter': self.maxiter}

        dt_max = np.inf
        dt_min = 1e-5

        if problem.__name__ == "run_piline":
            e_tol = 1e-7
            dt_min = 1e-2
        elif problem.__name__ == "run_vdp":
            e_tol = 2e-5
            dt_min = 1e-3
        elif problem.__name__ == "run_Lorenz":
            e_tol = 2e-5
            dt_min = 1e-3
        elif problem.__name__ == "run_Schroedinger":
            e_tol = 4e-6
            dt_min = 1e-3
        elif problem.__name__ == "run_quench":
            e_tol = 1e-5
            dt_min = 1e-3
            dt_max = 1e2
        else:
            raise NotImplementedError(
                'I don\'t have a tolerance for adaptivity for your problem. Please add one to the\
 strategy'
            )

        custom_description['level_params'] = {'restol': e_tol / 10 if self.restol is None else self.restol}
        custom_description['convergence_controllers'] = {
            AdaptivityCollocation: {
                'e_tol': e_tol,
                'dt_min': dt_min,
                'dt_max': dt_max,
                'adaptive_coll_params': self.adaptive_coll_params,
            }
        }
        return merge_descriptions(super().get_custom_description(problem, num_procs), custom_description)


class AdaptivityCollocationTypeStrategy(AdaptivityCollocationStrategy):
    def __init__(self, useMPI=False, skip_residual_computation='most'):
        super().__init__(useMPI=useMPI, skip_residual_computation=skip_residual_computation)
        self.color = list(cmap.values())[4]
        self.marker = '.'
        self.adaptive_coll_params = {
            'quad_type': ['RADAU-RIGHT', 'GAUSS'],
            'do_coll_update': [False, True],
        }

    @property
    def label(self):
        return 'adaptivity type'

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_vdp":
            if key == 'work_newton' and op == sum:
                return 2694
            elif key == 'e_global_post_run' and op == max:
                return 2.1707816100224875e-06

        raise NotImplementedError('The reference value you are looking for is not implemented for this strategy!')


class AdaptivityCollocationRefinementStrategy(AdaptivityCollocationStrategy):
    def __init__(self, useMPI=False, skip_residual_computation='most'):
        super().__init__(useMPI=useMPI, skip_residual_computation=skip_residual_computation)
        self.color = list(cmap.values())[5]
        self.marker = '^'
        self.adaptive_coll_params = {
            'num_nodes': [2, 3],
            'quad_type': ['GAUSS', 'RADAU-RIGHT'],
            'do_coll_update': [True, False],
        }

    @property
    def label(self):
        return 'adaptivity refinement'

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_vdp":
            if key == 'work_newton' and op == sum:
                return 1881
            elif key == 'e_global_post_run' and op == max:
                return 3.3428689164005654e-06

        raise NotImplementedError('The reference value you are looking for is not implemented for this strategy!')


class AdaptivityCollocationDerefinementStrategy(AdaptivityCollocationStrategy):
    def __init__(self, useMPI=False, skip_residual_computation='most'):
        super().__init__(useMPI=useMPI, skip_residual_computation=skip_residual_computation)
        self.color = list(cmap.values())[6]
        self.marker = '^'
        self.adaptive_coll_params = {'num_nodes': [4, 3]}

    @property
    def label(self):
        return 'adaptivity de-refinement'

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_vdp":
            if key == 'work_newton' and op == sum:
                return 3421
            elif key == 'e_global_post_run' and op == max:
                return 2.1130961994131336e-05

        raise NotImplementedError('The reference value you are looking for is not implemented for this strategy!')


class DIRKStrategy(AdaptivityStrategy):
    '''
    DIRK4(3)
    '''

    def __init__(self, useMPI=False, skip_residual_computation='all'):
        '''
        Initialization routine
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityRK

        super().__init__(useMPI=useMPI, skip_residual_computation=skip_residual_computation)
        self.color = list(cmap.values())[7]
        self.marker = '^'
        self.name = 'DIRK'
        self.bar_plot_x_label = 'DIRK4(3)'
        self.precision_parameter = 'e_tol'
        self.precision_parameter_loc = ['convergence_controllers', AdaptivityRK, 'e_tol']
        self.max_steps = 1e5

    @property
    def label(self):
        return 'DIRK4(3)'

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds adaptivity

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom descriptions you can supply to the problem when running it
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityRK, Adaptivity
        from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestarting
        from pySDC.implementations.sweeper_classes.Runge_Kutta import DIRK43

        adaptivity_description = super().get_custom_description(problem, num_procs)

        e_tol = adaptivity_description['convergence_controllers'][Adaptivity]['e_tol']
        adaptivity_description['convergence_controllers'].pop(Adaptivity, None)
        adaptivity_description.pop('sweeper_params', None)

        rk_params = {
            'step_params': {'maxiter': 1},
            'sweeper_class': DIRK43,
            'convergence_controllers': {
                AdaptivityRK: {'e_tol': e_tol},
                BasicRestarting.get_implementation(useMPI=self.useMPI): {
                    'max_restarts': 49,
                    'crash_after_max_restarts': False,
                },
            },
        }

        custom_description = merge_descriptions(adaptivity_description, rk_params)

        return custom_description

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_vdp":
            if key == 'work_newton' and op == sum:
                return 2168
            elif key == 'e_global_post_run' and op == max:
                return 0.00024166437265116247

        raise NotImplementedError('The reference value you are looking for is not implemented for this strategy!')

    def get_random_params(self, problem, num_procs):
        '''
        Routine to get parameters for the randomization of faults

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            dict: Randomization parameters
        '''
        rnd_params = super().get_random_params(problem, num_procs)
        rnd_params['iteration'] = 1
        rnd_params['min_node'] = 5

        return rnd_params


class ESDIRKStrategy(AdaptivityStrategy):
    '''
    ESDIRK5(3)
    '''

    def __init__(self, useMPI=False, skip_residual_computation='all'):
        '''
        Initialization routine
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityRK

        super().__init__(useMPI=useMPI, skip_residual_computation=skip_residual_computation)
        self.color = 'violet'
        self.marker = '^'
        self.name = 'ESDIRK'
        self.bar_plot_x_label = 'ESDIRK5(3)'
        self.precision_parameter = 'e_tol'
        self.precision_parameter_loc = ['convergence_controllers', AdaptivityRK, 'e_tol']
        self.max_steps = 1e5

    @property
    def label(self):
        return 'ESDIRK5(3)'

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds adaptivity

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom descriptions you can supply to the problem when running it
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityRK, Adaptivity
        from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestarting
        from pySDC.implementations.sweeper_classes.Runge_Kutta import ESDIRK53

        adaptivity_description = super().get_custom_description(problem, num_procs)

        e_tol = adaptivity_description['convergence_controllers'][Adaptivity]['e_tol']
        adaptivity_description['convergence_controllers'].pop(Adaptivity, None)
        adaptivity_description.pop('sweeper_params', None)

        rk_params = {
            'step_params': {'maxiter': 1},
            'sweeper_class': ESDIRK53,
            'convergence_controllers': {
                AdaptivityRK: {'e_tol': e_tol},
                BasicRestarting.get_implementation(useMPI=self.useMPI): {
                    'max_restarts': 49,
                    'crash_after_max_restarts': False,
                },
            },
        }

        custom_description = merge_descriptions(adaptivity_description, rk_params)

        return custom_description

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_vdp":
            if key == 'work_newton' and op == sum:
                return 1562
            elif key == 'e_global_post_run' and op == max:
                return 3.6982949243591356e-06

        raise NotImplementedError('The reference value you are looking for is not implemented for this strategy!')

    def get_random_params(self, problem, num_procs):
        '''
        Routine to get parameters for the randomization of faults

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            dict: Randomization parameters
        '''
        rnd_params = super().get_random_params(problem, num_procs)
        rnd_params['iteration'] = 1
        rnd_params['min_node'] = 6

        return rnd_params


class ERKStrategy(DIRKStrategy):
    """
    Explicit embedded RK using Cash-Karp's method
    """

    def __init__(self, useMPI=False, skip_residual_computation='all'):
        '''
        Initialization routine
        '''
        super().__init__(useMPI=useMPI, skip_residual_computation=skip_residual_computation)
        self.color = list(cmap.values())[9]
        self.marker = 'x'
        self.name = 'ERK'
        self.bar_plot_x_label = 'ERK5(4)'

    @property
    def label(self):
        return 'CP5(4)'

    def get_random_params(self, problem, num_procs):
        '''
        Routine to get parameters for the randomization of faults

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            dict: Randomization parameters
        '''
        rnd_params = super().get_random_params(problem, num_procs)
        rnd_params['min_node'] = 7

        return rnd_params

    def get_custom_description(self, problem, num_procs=1):
        from pySDC.implementations.sweeper_classes.Runge_Kutta import Cash_Karp

        desc = super().get_custom_description(problem, num_procs)
        desc['sweeper_class'] = Cash_Karp
        return desc

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_vdp":
            if key == 'work_newton' and op == sum:
                return 0
            elif key == 'e_global_post_run' and op == max:
                return 2.0606132165701396e-05

        raise NotImplementedError('The reference value you are looking for is not implemented for this strategy!')


class DoubleAdaptivityStrategy(AdaptivityStrategy):
    '''
    Adaptivity based both on embedded estimate and on residual
    '''

    def __init__(self, useMPI=False, skip_residual_computation='all'):
        '''
        Initialization routine
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

        super().__init__(useMPI=useMPI, skip_residual_computation=skip_residual_computation)
        self.color = list(cmap.values())[7]
        self.marker = '^'
        self.name = 'double_adaptivity'
        self.bar_plot_x_label = 'double adaptivity'
        self.precision_parameter = 'e_tol'
        self.precision_parameter_loc = ['convergence_controllers', Adaptivity, 'e_tol']
        self.residual_e_tol_ratio = 1.0
        self.residual_e_tol_abs = None

    @property
    def label(self):
        return 'double adaptivity'

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds adaptivity

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom descriptions you can supply to the problem when running it
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityResidual, Adaptivity
        from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestarting

        custom_description = super().get_custom_description(problem, num_procs)

        if self.residual_e_tol_abs:
            e_tol = self.residual_e_tol_abs
        else:
            e_tol = custom_description['convergence_controllers'][Adaptivity]['e_tol'] * self.residual_e_tol_ratio
        custom_description['convergence_controllers'][AdaptivityResidual] = {
            'e_tol': e_tol,
            'allowed_modifications': ['decrease'],
        }

        custom_description['convergence_controllers'][BasicRestarting.get_implementation(useMPI=self.useMPI)] = {
            'max_restarts': 15
        }

        return custom_description

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_vdp":
            if key == 'work_newton' and op == sum:
                return 3825
            elif key == 'e_global_post_run' and op == max:
                return 1.3370376368393444e-05

        raise NotImplementedError('The reference value you are looking for is not implemented for this strategy!')


class AdaptivityAvoidRestartsStrategy(AdaptivityStrategy):
    """
    Adaptivity with the avoid restarts option
    """

    @property
    def label(self):
        return 'adaptivity (avoid restarts)'

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds adaptivity

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom descriptions you can supply to the problem when running it
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
        from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestarting

        custom_description = super().get_custom_description(problem, num_procs)

        custom_description['convergence_controllers'][Adaptivity]['avoid_restarts'] = True

        custom_description['convergence_controllers'][BasicRestarting.get_implementation(useMPI=self.useMPI)] = {
            'max_restarts': 15
        }

        return custom_description

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_vdp":
            if key == 'work_newton' and op == sum:
                return 2955
            elif key == 'e_global_post_run' and op == max:
                return 5.274015506540053e-07

        raise NotImplementedError('The reference value you are looking for is not implemented for this strategy!')


class AdaptivityInterpolationStrategy(AdaptivityStrategy):
    """
    Adaptivity with interpolation between restarts
    """

    @property
    def label(self):
        return 'adaptivity+interpolation'

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds adaptivity

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom descriptions you can supply to the problem when running it
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
        from pySDC.implementations.convergence_controller_classes.interpolate_between_restarts import (
            InterpolateBetweenRestarts,
        )
        from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestarting

        custom_description = super().get_custom_description(problem, num_procs)

        custom_description['convergence_controllers'][Adaptivity]['avoid_restarts'] = False
        custom_description['convergence_controllers'][InterpolateBetweenRestarts] = {}

        custom_description['convergence_controllers'][BasicRestarting.get_implementation(useMPI=self.useMPI)] = {
            'max_restarts': 15
        }

        return custom_description

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_vdp":
            if key == 'work_newton' and op == sum:
                return 6659
            elif key == 'e_global_post_run' and op == max:
                return 2.9780002756552015e-06

        raise NotImplementedError('The reference value you are looking for is not implemented for this strategy!')


class AdaptivityExtrapolationWithinQStrategy(Strategy):
    '''
    Adaptivity based on extrapolation between collocation nodes as a resilience strategy
    '''

    def __init__(self, useMPI=False):
        '''
        Initialization routine
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityExtrapolationWithinQ

        super().__init__(useMPI=useMPI)
        self.color = list(cmap.values())[8]
        self.marker = '*'
        self.name = 'adaptivity_extraQ'
        self.bar_plot_x_label = 'adaptivity Q'
        self.precision_parameter = 'e_tol'
        self.adaptive_coll_params = {}
        self.precision_parameter_loc = ['convergence_controllers', AdaptivityExtrapolationWithinQ, 'e_tol']
        self.restol = None
        self.maxiter = 99

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds adaptivity

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom descriptions you can supply to the problem when running it
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityExtrapolationWithinQ

        custom_description = {}
        custom_description['step_params'] = {'maxiter': self.maxiter}

        dt_max = np.inf
        dt_min = 1e-5

        if problem.__name__ == "run_vdp":
            e_tol = 2e-5
            dt_min = 1e-3
        # elif problem.__name__ == "run_piline":
        #     e_tol = 1e-7
        #     dt_min = 1e-2
        # elif problem.__name__ == "run_Lorenz":
        #     e_tol = 2e-5
        #     dt_min = 1e-3
        # elif problem.__name__ == "run_Schroedinger":
        #     e_tol = 4e-6
        #     dt_min = 1e-3
        # elif problem.__name__ == "run_quench":
        #     e_tol = 1e-5
        #     dt_min = 1e-3
        #     dt_max = 1e2
        else:
            raise NotImplementedError(
                'I don\'t have a tolerance for adaptivity for your problem. Please add one to the\
 strategy'
            )

        if problem.__name__ in ['run_Schroedinger', 'run_quench']:
            from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

            sweeper_class = imex_1st_order
        else:
            from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit

            sweeper_class = generic_implicit

        custom_description['level_params'] = {'restol': e_tol / 10 if self.restol is None else self.restol}
        custom_description['convergence_controllers'] = {
            AdaptivityExtrapolationWithinQ: {
                'e_tol': e_tol,
                'dt_min': dt_min,
                'dt_max': dt_max,
            }
        }
        custom_description['sweeper_class'] = sweeper_class
        return merge_descriptions(super().get_custom_description(problem, num_procs), custom_description)

    def get_reference_value(self, problem, key, op, num_procs=1):
        """
        Get a reference value for a given problem for testing in CI.

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            key (str): The name of the variable you want to compare
            op (function): The operation you want to apply to the data
            num_procs (int): Number of processes

        Returns:
            The reference value
        """
        if problem.__name__ == "run_vdp":
            if key == 'work_newton' and op == sum:
                return 2259
            elif key == 'e_global_post_run' and op == max:
                return 9.319882663172407e-06

        raise NotImplementedError('The reference value you are looking for is not implemented for this strategy!')
