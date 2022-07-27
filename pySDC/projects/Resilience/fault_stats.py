import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

import pySDC.helpers.plot_helper as plot_helper
from pySDC.helpers.stats_helper import get_sorted

from fault_injection import FaultInjector
from pySDC.implementations.convergence_controller_classes.hotrod import HotRod
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

# these problems are available for testing
from pySDC.projects.Resilience.advection import run_advection
from pySDC.projects.Resilience.vdp import run_vdp
from pySDC.projects.Resilience.piline import run_piline

plot_helper.setup_mpl(reset=True)
cmap = TABLEAU_COLORS


class log_fault_stats_data(FaultInjector):
    '''
    This class stores all relevant information and allows fault injection
    '''

    def post_step(self, step, level_number):

        super(log_fault_stats_data, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='u', value=L.uend)
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='dt', value=L.dt)
        self.increment_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=0,
                             sweep=L.status.sweep, type='k', value=step.status.iter)
        self.increment_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=0,
                             sweep=L.status.sweep, type='restarts', value=int(step.status.restart))


class Strategy:
    '''
    Abstract class for resilience strategies
    '''
    def __init__(self):
        '''
        Initialization routine
        '''

        # set default values for plotting
        self.linestyle = '-'
        self.marker = '.'
        self.name = ''
        self.color = list(cmap.values())[0]

        # setup custom descriptions
        self.custom_description = {}

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that realizes the resilience strategy and taylors it to the problem at hand

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processses you intend to run with

        Returns:
            dict: The custom desciptions you can supply to the problem when running it
        '''

        return self.custom_description

    def get_fault_args(self, problem, num_procs):
        '''
        Routine to get arguments for the faults that are exempt from randomization

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processses you intend to run with

        Returns:
            dict: Arguments for the faults that are exempt from randomization
        '''

        return {}

    def get_random_params(self, problem, num_procs):
        '''
        Routine to get parameters for the randomization of faults

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processses you intend to run with

        Returns:
            dict: Randomization parameters
        '''

        return {}


class BaseStrategy(Strategy):
    '''
    Do a fixed iteration count
    '''
    def __init__(self):
        '''
        Initialization routine
        '''
        super(BaseStrategy, self).__init__()
        self.color = list(cmap.values())[0]
        self.marker = 'o'
        self.name = 'base'


class AdaptivityStrategy(Strategy):
    '''
    Adaptivity as a resilience strategy
    '''
    def __init__(self):
        '''
        Initialization routine
        '''
        super(AdaptivityStrategy, self).__init__()
        self.color = list(cmap.values())[1]
        self.marker = '*'
        self.name = 'adaptivity'

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds adaptivity

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processses you intend to run with

        Returns:
            The custom desciptions you can supply to the problem when running it
        '''
        if problem == run_piline:
            e_tol = 1e-7
            dt_min = 1e-2
        elif problem == run_vdp:
            e_tol = 3e-5
            dt_min = 1e-3
        else:
            raise NotImplementedError('I don\'t have a tolerance for adaptivity for your problem. Please add one to the\
 strategy')

        custom_description = {'convergence_controllers': {Adaptivity: {'e_tol': e_tol, 'dt_min': dt_min}}}

        return {**custom_description, **self.custom_description}


class IterateStrategy(Strategy):
    '''
    Iterate for as much as you want
    '''
    def __init__(self):
        '''
        Initialization routine
        '''
        super(IterateStrategy, self).__init__()
        self.color = list(cmap.values())[2]
        self.marker = 'v'
        self.name = 'iterate'

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that allows for adaptive iteration counts

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processses you intend to run with

        Returns:
            The custom desciptions you can supply to the problem when running it
        '''
        if problem == run_piline:
            restol = 2.3e-8
        elif problem == run_vdp:
            restol = 9e-7
        else:
            raise NotImplementedError('I don\'t have a residual tolerance for your problem. Please add one to the \
strategy')

        custom_description = {
            'step_params': {'maxiter': 99},
            'level_params': {'restol': restol},
        }

        return {**custom_description, **self.custom_description}


class HotRodStrategy(Strategy):
    '''
    Hot Rod as a resilience strategy
    '''
    def __init__(self):
        '''
        Initialization routine
        '''
        super(HotRodStrategy, self).__init__()
        self.color = list(cmap.values())[3]
        self.marker = '^'
        self.name = 'Hot Rod'

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds Hot Rod

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processses you intend to run with

        Returns:
            The custom desciptions you can supply to the problem when running it
        '''
        if problem == run_vdp:
            HotRod_tol = 5e-7
        else:
            raise NotImplementedError('I don\'t have a tolerance for Hot Rod for your problem. Please add one to the\
 strategy')

        no_storage = num_procs > 1

        custom_description = {'convergence_controllers': {HotRod: {'HotRod_tol': HotRod_tol, 'no_storage': no_storage}}}

        return {**custom_description, **self.custom_description}


class FaultStats:
    '''
    Class to generate and analyse fault statistics
    '''

    def __init__(self, prob=None, strategies=None, faults=None, reload=True, recovery_thresh=1 + 1e-3,
                 num_procs=1):
        '''
        Initialization routine

        Args:
            prob: A function that runs a pySDC problem, see imports for available problems
            strategies (list): List of resilience strategies
            faults (list): List of booleans that describe whether to use faults or not
            reload (bool): Load previously computed statisitics and continue from there or start from scratch
            recovery_thresh (float): Relative threshold for recovery
            num_procs (int): Number of processes
        '''
        self.prob = prob
        self.strategies = [None] if strategies is None else strategies
        self.faults = [False, True] if faults is None else faults
        self.reload = reload
        self.recovery_thresh = recovery_thresh
        self.num_procs = num_procs

    def get_Tend(self):
        '''
        Get the final time of runs for fault stats based on the problem

        Returns:
            float: Tend to put into the run
        '''
        if self.prob == run_vdp:
            return 2.3752559741400825
        elif self.prob == run_piline:
            return 20.
        else:
            raise NotImplementedError('I don\'t have a final time for your problem!')

    def get_custom_description(self):
        '''
        Get a custom description based on the problem

        Returns:
            dict: Custom description
        '''
        custom_description = {}
        if self.prob == run_vdp:
            custom_description['step_params'] = {'maxiter': 3}
        return custom_description

    def get_custom_problem_params(self):
        '''
        Get a custom problem parameters based on the problem

        Returns:
            dict: Custom problem params
        '''
        custom_params = {}
        if self.prob == run_vdp:
            custom_params = {
                'u0': np.array([0.99995, -0.00999985]),
                'crash_at_maxiter': False,
            }
        return custom_params

    def run_stats_generation(self, runs, step=None):
        '''
        Run the generation of stats for all strategies in the `self.strategies` variable

        Args:
            runs (int): Number of runs you want to do
            step (int): Number of runs you want to do between saving

        Returns:
            None
        '''
        step = runs if step is None else step
        reload = False

        for i in range(step, runs + step, step):
            for j in range(len(self.strategies)):
                for f in self.faults:
                    if f:
                        runs_partial = i
                    else:
                        runs_partial = min([5, i])
                    self.generate_stats(strategy=self.strategies[j], runs=runs_partial, faults=f,
                                        reload=self.reload or reload)
                self.get_recovered(self.strategies[j])
            reload = True

        return None

    def generate_stats(self, strategy=None, runs=1000, reload=True, faults=True):
        '''
        Generate statistics for recovery from bit flips
        -----------------------------------------------

        Every run is given a different random seed such that we have different faults and the results are then stored

        Args:
            strategy (Strategy): Resilience strategy
            runs (int): Number of runs you want to do
            reload (bool): Load previously computed statisitics and continue from there or start from scratch
            faults (bool): Whether to do stats with faults or whithout

        Returns:
            None
        '''

        # initialize dictionary to store the stats in
        dat = {
            'level': np.zeros(runs),
            'iteration': np.zeros(runs),
            'node': np.zeros(runs),
            'problem_pos': [],
            'bit': np.zeros(runs),
            'error': np.zeros(runs),
            'total_iteration': np.zeros(runs),
            'restarts': np.zeros(runs),
            'target': np.zeros(runs),
        }

        # reload previously recorded stats and write them to dat
        if reload:
            already_completed = self.load(strategy, faults)
            if already_completed['runs'] > 0 and already_completed['runs'] <= runs:
                for k in dat.keys():
                    dat[k][:min([already_completed['runs'], runs])] = already_completed.get(k, [])
        else:
            already_completed = {'runs': 0}

        if already_completed['runs'] < runs:
            if faults:
                print(f'Doing {strategy.name} with faults from {already_completed["runs"]} to {runs}')
            else:
                print(f'Doing {strategy.name} from {already_completed["runs"]} to {runs}')

        # perform the remaining experiments
        for i in range(already_completed['runs'], runs):

            # perform a single experiment with the correct random seed
            rng = np.random.RandomState(i)
            stats, controller, Tend = self.single_run(strategy, rng, faults)

            # get the data from the stats
            faults_run = get_sorted(stats, type='bitflip')
            t, u = get_sorted(stats, type='u', recomputed=False)[-1]

            # check if we ran to the end
            if t < Tend:
                error = np.inf
            else:
                error = abs(u - controller.MS[0].levels[0].prob.u_exact(t=t))
            total_iteration = sum([k[1] for k in get_sorted(stats, type='k')])

            # record the new data point
            if faults:
                assert len(faults_run) > 0, f'No faults where recorded in run {i} of strategy {strategy.name}!'
                dat['level'][i] = faults_run[0][1][0]
                dat['iteration'][i] = faults_run[0][1][1]
                dat['node'][i] = faults_run[0][1][2]
                dat['problem_pos'] += [faults_run[0][1][3]]
                dat['bit'][i] = faults_run[0][1][4]
                dat['target'][i] = faults_run[0][1][5]
            dat['error'][i] = error
            dat['total_iteration'][i] = total_iteration
            dat['restarts'][i] = sum([me[1] for me in get_sorted(stats, type='restart')])

        # store the completed stats
        dat['runs'] = runs
        if already_completed['runs'] < runs:
            self.store(strategy, faults, dat)

        return None

    def single_run(self, strategy, rng, faults, force_params=None):
        '''
        Run the problem once with the specified parameters

        Args:
            strategy (Strategy): The resilience strategy you plan on using
            rng (np.random.RandomState): Random generator to ensure repeatability
            faults (bool): Whether or not to put faults in
            force_params (dict): Change parameters in the description of the problem

        Returns:
            dict: Stats object containing statistics for each step, each level and each iteration
            pySDC.Controller: The controller of the run
            float: The time the problem should have run to
        '''
        force_params = {} if force_params is None else force_params

        # buid the custom description
        custom_description_prob = self.get_custom_description()
        custom_description_strategy = strategy.get_custom_description(self.prob, self.num_procs)
        custom_description = {**custom_description_prob, **custom_description_strategy}
        for k in force_params.keys():
            custom_description[k] = {**custom_description.get(k, {}), **force_params[k]}

        custom_controller_params = force_params.get('controller_params', {})
        custom_problem_params = self.get_custom_problem_params()

        if faults:
            fault_stuff = {
                'rng': rng,
                'args': strategy.get_fault_args(self.prob, self.num_procs),
                'rnd_params': strategy.get_fault_args(self.prob, self.num_procs),
            }
        else:
            fault_stuff = None

        return self.prob(custom_description=custom_description, num_procs=self.num_procs,
                         hook_class=log_fault_stats_data, fault_stuff=fault_stuff, Tend=self.get_Tend(),
                         custom_controller_params=custom_controller_params, custom_problem_params=custom_problem_params)

    def scrutinize(self, strategy, run, faults=True):
        '''
        Take a closer look at a specific run

        Args:
            strategy (Strategy): The resilience strategy you plan on using
            run (int): The number of the run to get the appropriate random generator
            faults (bool): Whether or not to include faults

        Returns:
            None
        '''
        rng = np.random.RandomState(run)

        force_params = dict()
        force_params['controller_params'] = {'logger_level': 15}

        stats, controller, Tend = self.single_run(strategy, rng, faults, force_params)

        t, u = get_sorted(stats, type='u')[-1]
        k_tot = sum([me[1] for me in get_sorted(stats, type='k')])

        # see if we can determine if the faults where recovered
        no_faults = self.load(strategy, False)
        e_star = np.mean(no_faults.get('error', [0]))
        error = abs(u - controller.MS[0].levels[0].prob.u_exact(t=t))
        recovery_thresh = e_star * self.recovery_thresh

        print(f'\ne={error:.2e}, e^*={e_star:.2e}, thresh: {recovery_thresh:.2e} -> recovered: \
{error < recovery_thresh}')
        print(f'k_tot={k_tot}')

        # checkout the step size
        dt = [me[1] for me in get_sorted(stats, type='dt')]
        print(f'dt: min: {np.min(dt):.2e}, max: {np.max(dt):.2e}, mean: {np.mean(dt):.2e}')

        # print faults
        faults = get_sorted(stats, type='bitflip')
        print('\nfaults:')
        print('   t  | level | iter | node | bit | trgt | pos')
        print('------+-------+------+------+-----+------+----')
        for f in faults:
            print(f' {f[0]:.2f} | {f[1][0]:5d} | {f[1][1]:4d} | {f[1][2]:4d} | {f[1][4]:3d} | {f[1][5]:4d} |', f[1][3])

        return None

    def convert_faults(self, faults):
        '''
        Make arrays of useable data from an entry in the stats object returned by pySDC

        Args:
            faults (list): The entry for faults returned by the pySDC run

        Returns:
            list: The times when the faults happened
            list: The levels in which the faults happened
            list: The iterations in which the faults happened
            list: The nodes in which the faults happened
            list: The problem positions in which the faults happened
            list: The bits in which the faults happened
        '''
        time = [faults[i][0] for i in range(len(faults))]
        level = [faults[i][1][0] for i in range(len(faults))]
        iteration = [faults[i][1][1] for i in range(len(faults))]
        node = [faults[i][1][2] for i in range(len(faults))]
        problem_pos = [faults[i][1][3] for i in range(len(faults))]
        bit = [faults[i][1][4] for i in range(len(faults))]
        return time, level, iteration, node, problem_pos, bit

    def get_path(self, strategy, faults):
        '''
        Get the path to where the stats are stored

        Args:
            strategy (Strategy): The resilience strategy
            faults (bool): Whether or not faults have been activated

        Returns:
            str: The path to what you are looking for
        '''
        return f'data/stats/{self.get_name(strategy, faults)}.pickle'

    def get_name(self, strategy=None, faults=False):
        '''
        Function to get a unique name for a kind of statistics based on the problem and strategy that was used

        Args:
            strategy (Strategy): Resilience strategy
            faults (bool): Whether or not faults where inserted

        Returns:
            str: The unique identifier
        '''
        if self.prob == run_advection:
            prob_name = 'advection'
        elif self.prob == run_vdp:
            prob_name = 'vdp'
        elif self.prob == run_piline:
            prob_name = 'piline'
        else:
            raise NotImplementedError(f'Name not implemented for problem {self.prob}')

        if faults:
            fault_name = '-faults'
        else:
            fault_name = ''

        if strategy is not None:
            strategy_name = f'-{strategy.name}'
        else:
            strategy_name = ''

        return f'{prob_name}{strategy_name}{fault_name}-{self.num_procs}procs'

    def store(self, strategy, faults, dat):
        '''
        Stores the data for a run at a predefined path

        Args:
            strategy (Strategy): Resilience strategy
            faults (bool): Whether or not faults where inserted
            dat (dict): The data of the recorded statistics

        Returns:
            None
        '''
        with open(self.get_path(strategy, faults), 'wb') as f:
            pickle.dump(dat, f)
        return None

    def load(self, strategy, faults):
        '''
        Loads the stats belonging to a specific strategy and whether or not faults where inserted.
        When no data has been generated yet, a dictionary is returned which only contains the number of completed runs,
        which is 0 of course.

        Args:
            strategy (Strategy): Resilience strategy
            faults (bool): Whether or not faults where inserted

        Returns:
            dict: Data from previous run or if it is not avalable a placeholder dictionary
        '''
        try:
            with open(self.get_path(strategy, faults), 'rb') as f:
                dat = pickle.load(f)
        except FileNotFoundError:
            return {'runs': 0}
        return dat

    def get_recovered(self, strategy):
        fault_free = self.load(strategy, False)
        with_faults = self.load(strategy, True)

        assert fault_free['error'].std() / fault_free['error'].mean() < 1e-5

        with_faults['recovered'] = with_faults['error'] < self.recovery_thresh * fault_free['error'].mean()
        self.store(strategy, True, with_faults)

    def crash_rate(self, dat, no_faults, thingA, mask):
        '''
        Determine the rate of runs that crashed

        Args:
            dat (dict): The data of the recorded statistics with faults
            no_faults (dict): The data of the corresponding fault-free stats
            thingA (str): Some key stored in the stats that will go on the y-axis
            mask (Numpy.ndarray of shape (n)): Arbitrary mask to apply before determining the rate

        Returns:
            int: Ratio of the runs which crashed and fall under the specific criteria set by the mask
        '''
        if len(dat[thingA][mask]) > 0:
            crash = dat['error'] == np.inf
            return len(dat[thingA][mask & crash]) / len(dat[thingA][mask])
        else:
            return None

    def rec_rate(self, dat, no_faults, thingA, mask):
        if len(dat[thingA][mask]) > 0:
            return len(dat[thingA][mask & dat['recovered']]) / len(dat[thingA][mask])
        else:
            return None

    def mean(self, dat, no_faults, thingA, mask):
        return np.mean(dat[thingA][mask])

    def extra_mean(self, dat, no_faults, thingA, mask):
        return np.mean(dat[thingA][mask]) - np.mean(no_faults[thingA])

    def plot_thingA_per_thingB(self, strategy, thingA, thingB, ax=None, mask=None, recovered=False, op=None):
        '''
        Plot thingA vs. thingB for a single strategy

        Args:
            strategy (Strategy): The resilience strategy you want to plot
            thingA (str): Some key stored in the stats that will go on the y-axis
            thingB (str): Some key stored in the stats that will go on the x-axis
            ax (Matplotlib.axes): Somewhere to plot
            mask (Numpy.ndarray of shape (n)): Arbitrary mask to apply to both axes
            recovered (bool): Show the plot for both all runs and only the recovered ones
            op (function): Operation that is applied to thingA before plotting default is recovery rate

        Returns:
            None
        '''
        op = self.rec_rate if op is None else op
        dat = self.load(strategy, True)
        no_faults = self.load(strategy, False)

        if mask is None:
            mask = np.ones_like(dat[thingB], dtype=bool)

        admissable_thingB = np.unique(dat[thingB][mask])
        me = np.zeros(len(admissable_thingB))
        me_recovered = np.zeros_like(me)

        for i in range(len(me)):
            _mask = (dat[thingB] == admissable_thingB[i]) & mask
            if _mask.any():
                me[i] = op(dat, no_faults, thingA, _mask)
                me_recovered[i] = op(dat, no_faults, thingA, _mask & dat['recovered'])

        if recovered:
            ax.plot(admissable_thingB, me_recovered, label=f'{strategy.name} (only recovered)', color=strategy.color,
                    marker=strategy.marker, ls='--', linewidth=3)

        ax.plot(admissable_thingB, me, label=f'{strategy.name}', color=strategy.color, marker=strategy.marker,
                linewidth=2)

        ax.legend(frameon=False)
        ax.set_xlabel(thingB)
        ax.set_ylabel(thingA)
        return None

    def plot_things_per_things(self, thingA, thingB, recovered=False, mask=None, op=None, args=None, strategies=None,
                               name=None, store=True, ax=None):
        '''
        Plot thingA vs thingB for multiple strategies

        Args:
            thingA (str): Some key stored in the stats that will go on the y-axis
            thingB (str): Some key stored in the stats that will go on the x-axis
            recovered (bool): Show the plot for both all runs and only the recovered ones
            mask (Numpy.ndarray of shape (n)): Arbitrary mask to apply to both axes
            op (function): Operation that is applied to thingA before plotting default is recovery rate
            args (dict): Parameters for how the plot should look
            strategies (list): List of the strategies you want to plot, if None, all will be plotted
            name (str): Optional name for the plot
            store (bool): Store the plot at a predefined path or not (for jupyter notebooks)
            ax (Matplotlib.axes): Somewhere to plot

        Returns
            None
        '''
        strategies = self.strategies if strategies is None else strategies
        args = {} if args is None else args

        # make sure we have something to plot in
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            store = False

        # execute the plots for all strategies
        for s in self.strategies:
            self.plot_thingA_per_thingB(s, thingA=thingA, thingB=thingB, recovered=recovered, ax=ax, mask=mask, op=op)

        # set the parameters
        [plt.setp(ax, k, v) for k, v in args.items()]

        if store:
            fig.tight_layout()
            plt.savefig(f'data/{self.get_name()}-{thingA if name is None else name}_per_{thingB}.pdf', transparent=True)
            plt.close(fig)

        return None

    def plot_recovery_thresholds(self, strategies=None, thresh_range=None, ax=None):
        # fill default values if nothing is specified
        strategies = self.strategies if strategies is None else strategies
        thresh_range = 1 + np.linspace(-4e-2, 4e-2, 100) if thresh_range is None else thresh_range
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        rec_rates = [[None] * len(thresh_range)] * len(strategies)
        for strategy_idx in range(len(strategies)):
            strategy = strategies[strategy_idx]
            # load the stats
            fault_free = self.load(strategy, False)
            with_faults = self.load(strategy, True)

            for thresh_idx in range(len(thresh_range)):
                rec_mask = with_faults['error'] < thresh_range[thresh_idx] * fault_free['error'].mean()
                rec_rates[strategy_idx][thresh_idx] = len(with_faults['error'][rec_mask]) / len(with_faults['error'])

            ax.plot(thresh_range, rec_rates[strategy_idx], color=strategy.color, label=strategy.name)
        ax.legend(frameon=False)
        ax.set_ylabel('recovery rate')
        ax.set_xlabel('threshold as ratio to fault-free error')

    def print_stats(self, strategy):
        dat = self.load(strategy, True)
        strings = {
            'iteration': 'iter',
            'node': 'nodes',
            'bit': 'bits',
        }
        print(f'Stats for {strategy.name}')
        for k, v in strings.items():
            me = np.unique(dat[k])
            for i in range(len(me)):
                mask = dat[k] == me[i]
                v = f'{v} {len(dat[k][mask])}'
            print(v)

    def get_mask(self, strategy=None, key=None, val=None, op='eq', old_mask=None):
        strategy = self.strategies[0] if strategy is None else strategy
        dat = self.load(strategy, True)
        if None in [key, val]:
            mask = dat['bit'] == dat['bit']
        else:
            if op == 'uneq':
                mask = dat[key] != val
            elif op == 'eq':
                mask = dat[key] == val
            elif op == 'leq':
                mask = dat[key] <= val
            elif op == 'geq':
                mask = dat[key] >= val
            elif op == 'lt':
                mask = dat[key] < val
            elif op == 'gt':
                mask = dat[key] > val
            else:
                raise NotImplementedError

        if old_mask is not None:
            return mask & old_mask
        else:
            return mask

    def get_index(self, mask):
        return np.arange(len(mask))[mask]


def main():
    stats_analyser = FaultStats(prob=run_vdp, strategies=[BaseStrategy(), AdaptivityStrategy(), IterateStrategy()],
                                faults=[False, True], reload=True, recovery_thresh=1 + 2e-2, num_procs=1)

    # stats_analyser.scrutinize(stats_analyser.strategies[2], 0, True)

    stats_analyser.run_stats_generation(runs=1000, step=50)
    mask = None

    stats_analyser.plot_things_per_things('recovered', 'node', False, op=stats_analyser.rec_rate, mask=mask,
                                          args={'ylabel': 'recovery rate'})
    stats_analyser.plot_things_per_things('recovered', 'iteration', False, op=stats_analyser.rec_rate, mask=mask,
                                          args={'ylabel': 'recovery rate'})
    stats_analyser.plot_things_per_things('recovered', 'bit', False, op=stats_analyser.rec_rate, mask=mask,
                                          args={'ylabel': 'recovery rate'})
    stats_analyser.plot_things_per_things('total_iteration', 'bit', True, op=stats_analyser.mean, mask=mask,
                                          args={'yscale': 'log', 'ylabel': 'total iterations'})
    stats_analyser.plot_things_per_things('total_iteration', 'bit', True, op=stats_analyser.extra_mean, mask=mask,
                                          args={'yscale': 'linear', 'ylabel': 'extra iterations'}, name='extra_iter')
    stats_analyser.plot_things_per_things('error', 'bit', True, op=stats_analyser.mean, mask=mask,
                                          args={'yscale': 'log'})

    s = 1
    mask = stats_analyser.get_mask(stats_analyser.strategies[s], 'iteration', 3, 'lt', mask)
    mask = stats_analyser.get_mask(stats_analyser.strategies[s], 'node', 0, 'gt', mask)
    mask = stats_analyser.get_mask(stats_analyser.strategies[s], 'recovered', False, 'eq', mask)
    print(stats_analyser.get_index(mask))

    stats_analyser.plot_recovery_thresholds()


if __name__ == "__main__":
    main()
