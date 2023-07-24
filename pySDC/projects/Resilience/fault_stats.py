import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpi4py import MPI

import pySDC.helpers.plot_helper as plot_helper
from pySDC.helpers.stats_helper import get_sorted

from pySDC.projects.Resilience.hook import hook_collection, LogUAllIter, LogData
from pySDC.projects.Resilience.fault_injection import get_fault_injector_hook
from pySDC.implementations.convergence_controller_classes.hotrod import HotRod
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
from pySDC.implementations.hooks.log_errors import LogLocalErrorPostStep
from pySDC.implementations.hooks.log_work import LogWork

# these problems are available for testing
from pySDC.projects.Resilience.advection import run_advection
from pySDC.projects.Resilience.vdp import run_vdp
from pySDC.projects.Resilience.piline import run_piline
from pySDC.projects.Resilience.Lorenz import run_Lorenz
from pySDC.projects.Resilience.Schroedinger import run_Schroedinger
from pySDC.projects.Resilience.quench import run_quench

from pySDC.projects.Resilience.strategies import BaseStrategy, AdaptivityStrategy, IterateStrategy, HotRodStrategy

plot_helper.setup_mpl(reset=True)


class FaultStats:
    '''
    Class to generate and analyse fault statistics
    '''

    def __init__(
        self,
        prob=None,
        strategies=None,
        faults=None,
        reload=True,
        recovery_thresh=1 + 1e-3,
        recovery_thresh_abs=0.0,
        num_procs=1,
        mode='combination',
        stats_path='data/stats',
        **kwargs,
    ):
        '''
        Initialization routine

        Args:
            prob: A function that runs a pySDC problem, see imports for available problems
            strategies (list): List of resilience strategies
            faults (list): List of booleans that describe whether to use faults or not
            reload (bool): Load previously computed statistics and continue from there or start from scratch
            recovery_thresh (float): Relative threshold for recovery
            num_procs (int): Number of processes
            mode (str): Mode for fault generation: Either 'random' or 'combination'
        '''
        self.prob = prob
        self.strategies = [None] if strategies is None else strategies
        self.faults = [False, True] if faults is None else faults
        self.reload = reload
        self.recovery_thresh = recovery_thresh
        self.recovery_thresh_abs = recovery_thresh_abs
        self.num_procs = num_procs
        self.mode = mode
        self.stats_path = stats_path
        self.kwargs = {
            'fault_frequency_iter': 500,
            **kwargs,
        }

    def get_Tend(self):
        '''
        Get the final time of runs for fault stats based on the problem

        Returns:
            float: Tend to put into the run
        '''
        return self.strategies[0].get_Tend(self.prob, self.num_procs)

    def run_stats_generation(self, runs=1000, step=None, comm=None, kwargs_range=None, _reload=False, _runs_partial=0):
        '''
        Run the generation of stats for all strategies in the `self.strategies` variable

        Args:
            runs (int): Number of runs you want to do
            step (int): Number of runs you want to do between saving
            comm (MPI.Communicator): Communicator for distributing runs
            kw_args_range (dict): Range for the parameters
            _reload, _runs_partial: Variables only used for recursion. Do not change!

        Returns:
            None
        '''
        for key, val in kwargs_range.items() if kwargs_range is not None else {}:
            if type(val) == int:
                self.kwargs[key] = val
            else:
                for me in val:
                    kwargs_range_me = {**kwargs_range, key: me}
                    self.run_stats_generation(runs=runs, step=step, comm=comm, kwargs_range=kwargs_range_me)
                return None

        comm = MPI.COMM_WORLD if comm is None else comm
        step = (runs if step is None else step) if comm.size == 1 else comm.size
        _runs_partial = step if _runs_partial == 0 else _runs_partial
        reload = self.reload or _reload

        max_runs = self.get_max_combinations() if self.mode == 'combination' else runs

        if reload:
            # sort the strategies to do some load balancing
            sorting_index = None
            if comm.rank == 0:
                already_completed = np.array(
                    [self.load(strategy=strategy, faults=True).get('runs', 0) for strategy in self.strategies]
                )
                sorting_index_ = np.argsort(already_completed)
                sorting_index = sorting_index_[already_completed[sorting_index_] < max_runs]

            # tell all ranks what strategies to use
            sorting_index = comm.bcast(sorting_index, root=0)
            strategies = [self.strategies[i] for i in sorting_index]
            if len(strategies) == 0:  # check if we are already done
                return None
        else:
            strategies = self.strategies

        strategy_comm = comm.Split(comm.rank % len(strategies))

        for j in range(0, len(strategies), comm.size):
            for f in self.faults:
                if f:
                    runs_partial = min(_runs_partial, max_runs)
                else:
                    runs_partial = min([5, _runs_partial])
                self.generate_stats(
                    strategy=strategies[j + (comm.rank % len(strategies) % (len(strategies)) - j)],
                    runs=runs_partial,
                    faults=f,
                    reload=reload,
                    comm=strategy_comm,
                )
        strategy_comm.Free()
        self.run_stats_generation(runs=runs, step=step, comm=comm, _reload=True, _runs_partial=_runs_partial + step)

        return None

    def generate_stats(self, strategy=None, runs=1000, reload=True, faults=True, comm=None):
        '''
        Generate statistics for recovery from bit flips
        -----------------------------------------------

        Every run is given a different random seed such that we have different faults and the results are then stored

        Args:
            strategy (Strategy): Resilience strategy
            runs (int): Number of runs you want to do
            reload (bool): Load previously computed statistics and continue from there or start from scratch
            faults (bool): Whether to do stats with faults or without
            comm (MPI.Communicator): Communicator for distributing runs

        Returns:
            None
        '''
        comm = MPI.COMM_WORLD if comm is None else comm

        # initialize dictionary to store the stats in
        dat = {
            'level': np.zeros(runs),
            'iteration': np.zeros(runs),
            'node': np.zeros(runs),
            'problem_pos': [],
            'bit': np.zeros(runs),
            'error': np.zeros(runs),
            'total_iteration': np.zeros(runs),
            'total_newton_iteration': np.zeros(runs),
            'restarts': np.zeros(runs),
            'target': np.zeros(runs),
        }

        # store arguments for storing and loading
        identifier_args = {
            'faults': faults,
            'strategy': strategy,
        }

        # reload previously recorded stats and write them to dat
        if reload:
            already_completed_ = None
            if comm.rank == 0:
                already_completed_ = self.load(**identifier_args)
            already_completed = comm.bcast(already_completed_, root=0)
            if already_completed['runs'] > 0 and already_completed['runs'] <= runs and comm.rank == 0:
                for k in dat.keys():
                    dat[k][: min([already_completed['runs'], runs])] = already_completed.get(k, [])
        else:
            already_completed = {'runs': 0}

        # prepare a message
        involved_ranks = comm.gather(MPI.COMM_WORLD.rank, root=0)
        msg = f'{comm.size} rank(s) ({involved_ranks}) doing {strategy.name}{" with faults" if faults else ""} from {already_completed["runs"]} to {runs}'
        if comm.rank == 0 and already_completed['runs'] < runs:
            print(msg, flush=True)

        space_comm = comm.Split(comm.rank)

        # perform the remaining experiments
        for i in range(already_completed['runs'], runs):
            if i % comm.size != comm.rank:
                continue

            # perform a single experiment with the correct random seed
            stats, controller, Tend = self.single_run(strategy=strategy, run=i, faults=faults, space_comm=space_comm)

            # get the data from the stats
            faults_run = get_sorted(stats, type='bitflip')
            t, u = get_sorted(stats, type='u', recomputed=False)[-1]

            # check if we ran to the end
            if t < Tend:
                error = np.inf
            else:
                error = self.get_error(u, t, controller, strategy)
            total_iteration = sum([k[1] for k in get_sorted(stats, type='k')])
            total_newton_iteration = sum([k[1] for k in get_sorted(stats, type='work_newton')])

            # record the new data point
            if faults:
                if len(faults_run) > 0:
                    dat['level'][i] = faults_run[0][1][0]
                    dat['iteration'][i] = faults_run[0][1][1]
                    dat['node'][i] = faults_run[0][1][2]
                    dat['problem_pos'] += [faults_run[0][1][3]]
                    dat['bit'][i] = faults_run[0][1][4]
                    dat['target'][i] = faults_run[0][1][5]
                else:
                    assert self.mode == 'regular', f'No faults where recorded in run {i} of strategy {strategy.name}!'
            dat['error'][i] = error
            dat['total_iteration'][i] = total_iteration
            dat['total_newton_iteration'][i] = total_newton_iteration
            dat['restarts'][i] = sum([me[1] for me in get_sorted(stats, type='restarts')])

        dat_full = {}
        for k in dat.keys():
            dat_full[k] = comm.reduce(dat[k], op=MPI.SUM)

        # store the completed stats
        dat_full['runs'] = runs

        if already_completed['runs'] < runs:
            if comm.rank == 0:
                self.store(dat_full, **identifier_args)
                if self.faults:
                    try:
                        self.get_recovered(strategy=strategy)
                    except KeyError:
                        print('Warning: Can\'t compute recovery rate right now')

        return None

    def get_error(self, u, t, controller, strategy):
        """
        Compute the error.

        Args:
            u (dtype_u): The solution at the end of the run
            t (float): Time at which `u` was recorded
            controller (pySDC.controller.controller): The controller
            strategy (Strategy): The resilience strategy

        Returns:
            float: Error
        """
        return abs(u - controller.MS[0].levels[0].prob.u_exact(t=t))

    def single_run(self, strategy, run=0, faults=False, force_params=None, hook_class=None, space_comm=None, Tend=None):
        '''
        Run the problem once with the specified parameters

        Args:
            strategy (Strategy): The resilience strategy you plan on using
            run (int): Index for fault generation
            faults (bool): Whether or not to put faults in
            force_params (dict): Change parameters in the description of the problem
            space_comm (MPI.Communicator): A communicator for space parallelisation

        Returns:
            dict: Stats object containing statistics for each step, each level and each iteration
            pySDC.Controller: The controller of the run
            float: The time the problem should have run to
        '''
        hook_class = hook_collection + [LogWork] + ([LogData] if hook_class is None else hook_class)
        force_params = {} if force_params is None else force_params

        # build the custom description
        custom_description = strategy.get_custom_description(self.prob, self.num_procs)
        for k in force_params.keys():
            custom_description[k] = {**custom_description.get(k, {}), **force_params[k]}

        custom_controller_params = force_params.get('controller_params', {})

        if faults:
            fault_stuff = {
                'rng': None,
                'args': strategy.get_fault_args(self.prob, self.num_procs),
                'rnd_params': strategy.get_fault_args(self.prob, self.num_procs),
            }

            # make parameters for faults:
            if self.mode == 'random':
                fault_stuff['rng'] = np.random.RandomState(run)
            elif self.mode == 'combination':
                fault_stuff['rng'] = run
            elif self.mode == 'regular':
                fault_stuff['rng'] = np.random.RandomState(run)
                fault_stuff['fault_frequency_iter'] = self.kwargs['fault_frequency_iter']
                fault_stuff['rnd_params'] = {
                    'bit': 12,
                    'min_node': 1,
                }
            else:
                raise NotImplementedError(f'Don\'t know how to add faults in mode {self.mode}')

        else:
            fault_stuff = None

        return self.prob(
            custom_description=custom_description,
            num_procs=self.num_procs,
            hook_class=hook_class,
            fault_stuff=fault_stuff,
            Tend=self.get_Tend() if Tend is None else Tend,
            custom_controller_params=custom_controller_params,
            space_comm=space_comm,
        )

    def compare_strategies(self, run=0, faults=False, ax=None):  # pragma: no cover
        '''
        Take a closer look at how the strategies compare for a specific run

        Args:
            run (int): The number of the run to get the appropriate random generator
            faults (bool): Whether or not to include faults
            ax (Matplotlib.axes): Somewhere to plot

        Returns:
            None
        '''
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            store = True
        else:
            store = False

        k_ax = ax.twinx()
        ls = ['-.' if type(strategy) == HotRodStrategy else '-' for strategy in self.strategies]
        [self.scrutinize_visual(self.strategies[i], run, faults, ax, k_ax, ls[i]) for i in range(len(self.strategies))]

        # make a legend
        [k_ax.plot([None], [None], label=strategy.label, color=strategy.color) for strategy in self.strategies]
        k_ax.legend(frameon=True)

        if store:
            fig.tight_layout()
            plt.savefig(f'data/{self.get_name()}-comparison.pdf', transparent=True)

    def scrutinize_visual(
        self, strategy, run, faults, ax=None, k_ax=None, ls='-', plot_restarts=False
    ):  # pragma: no cover
        '''
        Take a closer look at a specific run with a plot

        Args:
            strategy (Strategy): The resilience strategy you plan on using
            run (int): The number of the run to get the appropriate random generator
            faults (bool): Whether or not to include faults
            ax (Matplotlib.axes): Somewhere to plot the error
            k_ax (Matplotlib.axes): Somewhere to plot the iterations
            plot_restarts (bool): Make vertical lines wherever restarts happened

        Returns:
            None
        '''
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            store = True
        else:
            store = False

        force_params = {}

        stats, controller, Tend = self.single_run(
            strategy=strategy,
            run=run,
            faults=faults,
            force_params=force_params,
            hook_class=hook_collection + [LogLocalErrorPostStep, LogData],
        )

        # plot the local error
        e_loc = get_sorted(stats, type='e_local_post_step', recomputed=False)
        ax.plot([me[0] for me in e_loc], [me[1] for me in e_loc], color=strategy.color, ls=ls)

        # plot the iterations
        k_ax = ax.twinx() if k_ax is None else k_ax
        k = get_sorted(stats, type='k')
        k_ax.plot([me[0] for me in k], np.cumsum([me[1] for me in k]), color=strategy.color, ls='--')

        # plot the faults
        faults = get_sorted(stats, type='bitflip')
        for fault_time in [me[0] for me in faults]:
            ax.axvline(fault_time, color='grey', ls=':')

        # plot restarts
        if plot_restarts:
            restarts = get_sorted(stats, type='restarts')
            [ax.axvline(me[0], color='black', ls='-.') if me[1] else '' for me in restarts]

        # decorate
        ax.set_yscale('log')
        ax.set_ylabel(r'$\epsilon$')
        k_ax.set_ylabel('cumulative iterations (dashed)')
        ax.set_xlabel(r'$t$')

        if store:
            fig.tight_layout()
            plt.savefig(f'data/{self.get_name()}-{strategy.name}-details.pdf', transparent=True)

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
        force_params = {}
        force_params['controller_params'] = {'logger_level': 15}

        stats, controller, Tend = self.single_run(strategy=strategy, run=run, faults=faults, force_params=force_params)

        t, u = get_sorted(stats, type='u')[-1]
        print(max(u))
        k = [me[1] for me in get_sorted(stats, type='k')]
        print(k)

        print(f'\nOverview for {strategy.name} strategy')

        # see if we can determine if the faults where recovered
        no_faults = self.load(strategy=strategy, faults=False)
        e_star = np.mean(no_faults.get('error', [0]))
        if t < Tend:
            error = np.inf
            print(f'Final time was not reached! Code crashed at t={t:.2f} instead of reaching Tend={Tend:.2f}')
        else:
            error = self.get_error(u, t, controller, strategy)
        recovery_thresh = self.get_thresh(strategy)

        print(
            f'e={error:.2e}, e^*={e_star:.2e}, thresh: {recovery_thresh:.2e} -> recovered: \
{error < recovery_thresh}'
        )
        print(f'k: sum: {np.sum(k)}, min: {np.min(k)}, max: {np.max(k)}, mean: {np.mean(k):.2f},')

        _newton_iter = get_sorted(stats, type='work_newton')
        if len(_newton_iter) > 0:
            newton_iter = [me[1] for me in _newton_iter]
            print(
                f'Newton: k: sum: {np.sum(newton_iter)}, min: {np.min(newton_iter)}, max: {np.max(newton_iter)}, mean: {np.mean(newton_iter):.2f},'
            )

        # checkout the step size
        dt = [me[1] for me in get_sorted(stats, type='dt')]
        print(f'dt: min: {np.min(dt):.2e}, max: {np.max(dt):.2e}, mean: {np.mean(dt):.2e}')

        # restarts
        restarts = [me[1] for me in get_sorted(stats, type='restarts')]
        print(f'restarts: {sum(restarts)}, without faults: {no_faults["restarts"][0]}')

        # print faults
        faults = get_sorted(stats, type='bitflip')
        print('\nfaults:')
        print('    t   | level | iter | node | bit | trgt | pos')
        print('--------+-------+------+------+-----+------+----')
        for f in faults:
            print(f' {f[0]:6.2f} | {f[1][0]:5d} | {f[1][1]:4d} | {f[1][2]:4d} | {f[1][4]:3d} | {f[1][5]:4d} |', f[1][3])

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

    def get_path(self, **kwargs):
        '''
        Get the path to where the stats are stored

        Args:
            strategy (Strategy): The resilience strategy
            faults (bool): Whether or not faults have been activated

        Returns:
            str: The path to what you are looking for
        '''
        return f'{self.stats_path}/{self.get_name(**kwargs)}.pickle'

    def get_name(self, strategy=None, faults=True, mode=None):
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
        elif self.prob == run_Lorenz:
            prob_name = 'Lorenz'
        elif self.prob == run_Schroedinger:
            prob_name = 'Schroedinger'
        elif self.prob == run_quench:
            prob_name = 'Quench'
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

        mode = self.mode if mode is None else mode
        if mode == 'regular':
            mode_thing = f'-regular{self.kwargs["fault_frequency_iter"] if faults else ""}'
        else:
            mode_thing = ''

        return f'{prob_name}{strategy_name}{fault_name}-{self.num_procs}procs{mode_thing}'

    def store(self, dat, **kwargs):
        '''
        Stores the data for a run at a predefined path

        Args:
            dat (dict): The data of the recorded statistics

        Returns:
            None
        '''
        with open(self.get_path(**kwargs), 'wb') as f:
            pickle.dump(dat, f)
        return None

    def load(self, **kwargs):
        '''
        Loads the stats belonging to a specific strategy and whether or not faults where inserted.
        When no data has been generated yet, a dictionary is returned which only contains the number of completed runs,
        which is 0 of course.

        Returns:
            dict: Data from previous run or if it is not available a placeholder dictionary
        '''
        kwargs['strategy'] = kwargs.get('strategy', self.strategies[MPI.COMM_WORLD.rank % len(self.strategies)])

        try:
            with open(self.get_path(**kwargs), 'rb') as f:
                dat = pickle.load(f)
        except FileNotFoundError:
            return {'runs': 0}
        return dat

    def get_thresh(self, strategy=None):
        """
        Get recovery threshold based on relative and absolute tolerances

        Args:
            strategy (Strategy): The resilience strategy
        """
        fault_free = self.load(strategy=strategy, faults=False)
        assert fault_free['error'].std() / fault_free['error'].mean() < 1e-5
        return self.recovery_thresh_abs + self.recovery_thresh * fault_free["error"].mean()

    def get_recovered(self, **kwargs):
        '''
        Determine the recovery rate for a specific strategy and store it to disk.

        Returns:
            None
        '''
        if 'strategy' not in kwargs.keys():
            [self.get_recovered(strategy=strat, **kwargs) for strat in self.strategies]
        else:
            try:
                with_faults = self.load(faults=True, **kwargs)
                with_faults['recovered'] = with_faults['error'] < self.get_thresh(kwargs['strategy'])
                self.store(faults=True, dat=with_faults, **kwargs)
            except KeyError:
                print("Can\'t compute recovery rate right now")

        return None

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
        '''
        Operation for plotting which returns the recovery rate for a given mask.
        Which thingA you apply this to actually does not matter here since we compute a rate.

        Args:
            dat (dict): The recorded statistics
            no_faults (dict): The corresponding fault-free stats
            thingA (str): Some key stored in the stats
            mask (Numpy.ndarray of shape (n)): Arbitrary mask for filtering

        Returns:
            float: Recovery rate
        '''
        if len(dat[thingA][mask]) > 0:
            return len(dat[thingA][mask & dat['recovered']]) / len(dat[thingA][mask])
        else:
            return None

    def mean(self, dat, no_faults, thingA, mask):
        '''
        Operation for plotting which returns the mean of thingA after applying the mask

        Args:
            dat (dict): The recorded statistics
            no_faults (dict): The corresponding fault-free stats
            thingA (str): Some key stored in the stats
            mask (Numpy.ndarray of shape (n)): Arbitrary mask for filtering

        Returns:
            float: Mean of thingA after applying mask
        '''
        return np.mean(dat[thingA][mask])

    def extra_mean(self, dat, no_faults, thingA, mask):
        '''
        Operation for plotting which returns the difference in mean of thingA between runs with and without faults after
        applying the mask

        Args:
            dat (dict): The recorded statistics
            no_faults (dict): The corresponding fault-free stats
            thingA (str): Some key stored in the stats
            mask (Numpy.ndarray of shape (n)): Arbitrary mask for filtering

        Returns:
            float: Difference in mean of thingA between runs with and without faults after applying mask
        '''
        if True in mask or int in [type(me) for me in mask]:
            return np.mean(dat[thingA][mask]) - np.mean(no_faults[thingA])
        else:
            return None

    def plot_thingA_per_thingB(
        self, strategy, thingA, thingB, ax=None, mask=None, recovered=False, op=None
    ):  # pragma: no cover
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
        dat = self.load(strategy=strategy, faults=True)
        no_faults = self.load(strategy=strategy, faults=False)

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
            ax.plot(
                admissable_thingB,
                me_recovered,
                label=f'{strategy.label} (only recovered)',
                color=strategy.color,
                marker=strategy.marker,
                ls='--',
                linewidth=3,
            )

        ax.plot(
            admissable_thingB, me, label=f'{strategy.label}', color=strategy.color, marker=strategy.marker, linewidth=2
        )

        ax.legend(frameon=False)
        ax.set_xlabel(thingB)
        ax.set_ylabel(thingA)
        return None

    def plot_things_per_things(
        self,
        thingA='bit',
        thingB='bit',
        recovered=False,
        mask=None,
        op=None,
        args=None,
        strategies=None,
        name=None,
        store=True,
        ax=None,
        fig=None,
    ):  # pragma: no cover
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
            fig (Matplotlib.figure): Figure of the ax

        Returns
            None
        '''
        strategies = self.strategies if strategies is None else strategies
        args = {} if args is None else args

        # make sure we have something to plot in
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        elif fig is None:
            store = False

        # execute the plots for all strategies
        for s in strategies:
            self.plot_thingA_per_thingB(s, thingA=thingA, thingB=thingB, recovered=recovered, ax=ax, mask=mask, op=op)

        # set the parameters
        [plt.setp(ax, k, v) for k, v in args.items()]

        if store:
            fig.tight_layout()
            plt.savefig(f'data/{self.get_name()}-{thingA if name is None else name}_per_{thingB}.pdf', transparent=True)
            plt.close(fig)

        return None

    def plot_recovery_thresholds(self, strategies=None, thresh_range=None, ax=None):  # pragma: no cover
        '''
        Plot the recovery rate for a range of thresholds

        Args:
            strategies (list): List of the strategies you want to plot, if None, all will be plotted
            thresh_range (list): thresholds for deciding whether to accept as recovered
            ax (Matplotlib.axes): Somewhere to plot

        Returns:
            None
        '''
        # fill default values if nothing is specified
        strategies = self.strategies if strategies is None else strategies
        thresh_range = 1 + np.linspace(-4e-2, 4e-2, 100) if thresh_range is None else thresh_range
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        rec_rates = [[None] * len(thresh_range)] * len(strategies)
        for strategy_idx in range(len(strategies)):
            strategy = strategies[strategy_idx]
            # load the stats
            fault_free = self.load(strategy=strategy, faults=False)
            with_faults = self.load(strategy=strategy, faults=True)

            for thresh_idx in range(len(thresh_range)):
                rec_mask = with_faults['error'] < thresh_range[thresh_idx] * fault_free['error'].mean()
                rec_rates[strategy_idx][thresh_idx] = len(with_faults['error'][rec_mask]) / len(with_faults['error'])

            ax.plot(thresh_range, rec_rates[strategy_idx], color=strategy.color, label=strategy.label)
        ax.legend(frameon=False)
        ax.set_ylabel('recovery rate')
        ax.set_xlabel('threshold as ratio to fault-free error')

        return None

    def analyse_adaptivity(self, mask):  # pragma: no cover
        '''
        Analyse a set of runs with adaptivity

        Args:
            mask (Numpy.ndarray of shape (n)): The mask you want to know about

        Returns:
            None
        '''
        index = self.get_index(mask)
        dat = self.load()

        # make a header
        print('  run  | bit | node | iter |  e_em^*  |   e_em   | e_glob^* |  e_glob  ')
        print('-------+-----+------+------+----------+----------+----------+----------')
        for i in index:
            e_em, e_glob = self.analyse_adaptivity_single(int(i))
            print(
                f' {i:5d} | {dat["bit"][i]:3.0f} | {dat["node"][i]:4.0f} | {dat["iteration"][i]:4.0f} | {e_em[1]:.2e}\
 | {e_em[0]:.2e} | {e_glob[1]:.2e} | {e_glob[0]:.2e}'
            )

        e_tol = AdaptivityStrategy().get_custom_description(self.prob, self.num_procs)['convergence_controllers'][
            Adaptivity
        ]['e_tol']
        print(f'We only restart when e_em > e_tol = {e_tol:.2e}!')
        return None

    def analyse_adaptivity_single(self, run):  # pragma: no cover
        '''
        Compute what the difference in embedded and global error are for a specific run with adaptivity

        Args:
            run (int): The run you want to know about

        Returns:
            list: Embedded error with fault and without for the last iteration in the step with a fault
            list: Global error with and without fault at the end of the run
        '''
        # perform one run with and one without faults
        stats = []
        controllers = []
        for faults in [True, False]:
            s, c, _ = self.single_run(
                strategy=AdaptivityStrategy(), run=run, faults=faults, hook_class=hook_collection + [LogUAllIter]
            )
            stats += [s]
            controllers += [c]

        # figure out when the fault happened
        t_fault = get_sorted(stats[0], type='bitflip')[0][0]

        # get embedded error
        e_em = [
            [me[1] for me in get_sorted(stat, type='error_embedded_estimate', time=t_fault, sortby='iter')]
            for stat in stats
        ]

        # compute the global error
        u_end = [get_sorted(stat, type='u')[-1] for stat in stats]
        e_glob = [abs(u_end[i][1] - controllers[i].MS[0].levels[0].prob.u_exact(t=u_end[i][0])) for i in [0, 1]]

        return [e_em[i][-1] for i in [0, 1]], e_glob

    def analyse_HotRod(self, mask):  # pragma: no cover
        '''
        Analyse a set of runs with Hot Rod

        Args:
            mask (Numpy.ndarray of shape (n)): The mask you want to know about

        Returns:
            None
        '''
        index = self.get_index(mask)
        dat = self.load()

        # make a header
        print(
            '  run  | bit | node | iter |  e_ex^*  |   e_ex   |  e_em^*  |   e_em   |   diff*  |   diff   | e_glob^* \
|  e_glob  '
        )
        print(
            '-------+-----+------+------+----------+----------+----------+----------+----------+----------+----------\
+----------'
        )
        for i in index:
            e_em, e_ex, e_glob = self.analyse_HotRod_single(int(i))
            print(
                f' {i:5d} | {dat["bit"][i]:3.0f} | {dat["node"][i]:4.0f} | {dat["iteration"][i]:4.0f} | {e_ex[1]:.2e}\
 | {e_ex[0]:.2e} | {e_em[1]:.2e} | {e_em[0]:.2e} | {abs(e_em[1]-e_ex[1]):.2e} | {abs(e_em[0]-e_ex[0]):.2e} | \
{e_glob[1]:.2e} | {e_glob[0]:.2e}'
            )

        tol = HotRodStrategy().get_custom_description(self.prob, self.num_procs)['convergence_controllers'][HotRod][
            'HotRod_tol'
        ]
        print(f'We only restart when diff > tol = {tol:.2e}!')
        return None

    def analyse_HotRod_single(self, run):  # pragma: no cover
        '''
        Compute what the difference in embedded, extrapolated and global error are for a specific run with Hot Rod

        Args:
            run (int): The run you want to know about

        Returns:
            list: Embedded error with fault and without for the last iteration in the step with a fault
            list: Extrapolation error with fault and without for the last iteration in the step with a fault
            list: Global error with and without fault at the end of the run
        '''
        # perform one run with and one without faults
        stats = []
        controllers = []
        for faults in [True, False]:
            s, c, _ = self.single_run(
                strategy=HotRodStrategy(), run=run, faults=faults, hook_class=hook_collection + [LogUAllIter]
            )
            stats += [s]
            controllers += [c]

        # figure out when the fault happened
        t_fault = get_sorted(stats[0], type='bitflip')[0][0]

        # get embedded error
        e_em = [
            [me[1] for me in get_sorted(stat, type='error_embedded_estimate', time=t_fault, sortby='iter')]
            for stat in stats
        ]
        # get extrapolated error
        e_ex = [
            [me[1] for me in get_sorted(stat, type='error_extrapolation_estimate', time=t_fault, sortby='iter')]
            for stat in stats
        ]

        # compute the global error
        u_end = [get_sorted(stat, type='u')[-1] for stat in stats]
        e_glob = [abs(u_end[i][1] - controllers[i].MS[0].levels[0].prob.u_exact(t=u_end[i][0])) for i in [0, 1]]

        return [e_em[i][-1] for i in [0, 1]], [e_ex[i][-1] for i in [0, 1]], e_glob

    def print_faults(self, mask=None):  # pragma: no cover
        '''
        Print all faults that happened within a certain mask

        Args:
            mask (Numpy.ndarray of shape (n)): The mask you want to know the contents of

        Returns:
            None
        '''
        index = self.get_index(mask)
        dat = self.load()

        # make a header
        print('  run  | bit | node | iter | space pos')
        print('-------+-----+------+------+-----------')
        for i in index:
            print(
                f' {i:5d} | {dat["bit"][i]:3.0f} | {dat["node"][i]:4.0f} | {dat["iteration"][i]:4.0f} | \
{dat["problem_pos"][i]}'
            )

        return None

    def get_mask(self, strategy=None, key=None, val=None, op='eq', old_mask=None, compare_faults=False):
        '''
        Make a mask to apply to stored data to filter out certain things

        Args:
            strategy (Strategy): The resilience strategy you want to apply the mask to. Most masks are the same for all
                                 strategies so None is fine
            key (str): The key in the stored statistics that you want to filter for some value
            val (str, float, int, bool): A value that you want to use for filtering. Dtype depends on key
            op (str): Operation that is applied for filtering
            old_mask (Numpy.ndarray of shape (n)): Apply this mask on top of the filter
            compare_faults (bool): instead of comparing to val, compare to the mean value for fault free runs

        Returns:
            Numpy.ndarray with boolean entries that can be used as a mask
        '''
        strategy = self.strategies[0] if strategy is None else strategy
        dat = self.load(strategy=strategy, faults=True)

        if compare_faults:
            if val is not None:
                raise ValueError('Can\'t use val and compare_faults in get_mask at the same time!')
            else:
                vals = self.load(strategy=strategy, faults=False)[key]
                val = sum(vals) / len(vals)

        if None in [key, val] and op not in ['isfinite']:
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
            elif op == 'isfinite':
                mask = np.isfinite(dat[key])
            else:
                raise NotImplementedError(f'Please implement op={op}!')

        if old_mask is not None:
            return mask & old_mask
        else:
            return mask

    def get_fixable_faults_only(self, strategy):
        """
        Return a mask of only faults that can be fixed with a given strategy.

        Args:
            strategy (Strategy): The resilience strategy you want to look at. In normal use it's the same for all

        Returns:
            Numpy.ndarray with boolean entries that can be used as a mask
        """
        fixable = strategy.get_fixable_params(
            maxiter=strategy.get_custom_description(self.prob, self.num_procs)['step_params']['maxiter']
        )
        mask = self.get_mask(strategy=strategy)

        for kwargs in fixable:
            mask = self.get_mask(strategy=strategy, **kwargs, old_mask=mask)

        return mask

    def get_index(self, mask=None):
        '''
        Get the indeces of all runs in mask

        Args:
            mask (Numpy.ndarray of shape (n)): The mask you want to know the contents of

        Returns:
            Numpy.ndarray: Array of indeces
        '''
        if mask is None:
            dat = self.load()
            return np.arange(len(dat['iteration']))
        else:
            return np.arange(len(mask))[mask]

    def get_statistics_info(self, mask=None, strategy=None, print_all=False, ax=None):  # pragma: no cover
        '''
        Get information about how many data points for faults we have given a particular mask

        Args:
            mask (Numpy.ndarray of shape (n)): The mask you want to apply before counting
            strategy (Strategy): The resilience strategy you want to look at. In normal use it's the same for all
                                 strategies, so you don't need to supply this argument
            print_all (bool): Whether to add information that is normally not useful to the table
            ax (Matplotlib.axes): Somewhere to plot the combinations histogram

        Returns:
            None
        '''

        # load some data from which to infer the number occurrences of some event
        strategy = self.strategies[0] if strategy is None else strategy
        dat = self.load(stratagy=strategy, faults=True)

        # make a dummy mask in case none is supplied
        if mask is None:
            mask = np.ones_like(dat['error'], dtype=bool)

        # print a header
        print(f' tot: {len(dat["error"][mask]):6} | avg. counts | mean deviation | unique entries')
        print('-------------------------------------------------------------')

        # make a list of all keys that you want to look at
        keys = ['iteration', 'bit', 'node']
        if print_all:
            keys += ['problem_pos', 'level', 'target']

        # print the table
        for key in keys:
            counts, dev, unique = self.count_occurrences(dat[key][mask])
            print(f' {key:11} | {counts:11.1f} | {dev:14.2f} | {unique:14}')

        return None

    def combinations_histogram(self, dat=None, keys=None, mask=None, ax=None):  # pragma: no cover
        '''
        Make a histogram ouf of the occurrences of combinations

        Args:
            dat (dict): The data of the recorded statistics
            keys (list): The keys in dat that you want to know the combinations of
            mask (Numpy.ndarray of shape (n)): The mask you want to apply before counting

        Returns:
            Matplotlib.axes: The plot
        '''
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        occurrences, bins = self.get_combination_histogram(dat, keys, mask)

        ax.bar(bins[:-1], occurrences)

        ax.set_xlabel('Occurrence of combinations')

        return ax

    def get_combination_histogram(self, dat=None, keys=None, mask=None):  # pragma: no cover
        '''
        Check how many combinations of values we expect and how many we find to see if we need to do more experiments.
        It is assumed that each allowed value for each key appears at least once in dat after the mask was applied

        Args:
            dat (dict): The data of the recorded statistics
            keys (list): The keys in dat that you want to know the combinations of
            mask (Numpy.ndarray of shape (n)): The mask you want to apply before counting

        Returns:
            Numpy.ndarray: Number of occurrences of combinations
            Numpy.ndarray: Bins
        '''

        # load default values
        dat = self.load(strategy=self.strategies[0], faults=True) if dat is None else dat
        keys = ['iteration', 'bit', 'node'] if keys is None else keys
        if mask is None:
            mask = np.ones_like(dat['error'], dtype=bool)

        # get unique values and compute how many combinations you expect
        num_unique = [len(np.unique(dat[key][mask])) for key in keys]
        expected_number_of_combinations = np.prod(num_unique)

        # test what you actually get
        combination_counts = self.get_combination_counts(dat, keys, mask)

        # make a histogram with the result
        occurrences, bins = np.histogram(combination_counts, bins=np.arange(max(combination_counts) + 1))
        occurrences[0] = expected_number_of_combinations - len(combination_counts)

        return occurrences, bins

    def get_max_combinations(self, dat=None):
        '''
        Count how many combinations of parameters for faults are possible

        Args:
            dat (dict): The recorded statistics
            keys (list): The keys in dat that you want to know the combinations of

        Returns:
            int: Number of possible combinations
        '''
        stats, controller, Tend = self.single_run(strategy=self.strategies[0], run=0, faults=True)
        faultHook = get_fault_injector_hook(controller)
        ranges = [
            (0, faultHook.rnd_params['level_number']),
            (0, faultHook.rnd_params['node'] + 1),
            (1, faultHook.rnd_params['iteration'] + 1),
            (0, faultHook.rnd_params['bit']),
        ]
        ranges += [(0, i) for i in faultHook.rnd_params['problem_pos']]
        return np.prod([me[1] - me[0] for me in ranges], dtype=int)

    def get_combination_counts(self, dat, keys, mask):
        '''
        Get counts of how often all combinations of values of keys appear. This is done recursively to support arbitrary
        numbers of keys

        Args:
            dat (dict): The data of the recorded statistics
            keys (list): The keys in dat that you want to know the combinations of
            mask (Numpy.ndarray of shape (n)): The mask you want to apply before counting

        Returns:
            list: Occurrences of all combinations
        '''
        key = keys[0]
        unique_vals = np.unique(dat[key][mask])
        res = []

        for i in range(len(unique_vals)):
            inner_mask = self.get_mask(key=key, val=unique_vals[i], op='eq', old_mask=mask)
            if len(keys) > 1:
                res += self.get_combination_counts(dat, keys[1:], inner_mask)
            else:
                res += [self.count_occurrences(dat[key][inner_mask])[0]]
        return res

    def count_occurrences(self, vals):
        '''
        Count the occurrences of unique values in vals and compute average deviation from mean

        Args:
            vals (list): Values you want to check

        Returns:
            float: Mean of number of occurrences of unique values in vals
            float: Average deviation from mean number of occurrences
            int: Number of unique entries
        '''
        unique_vals, counts = np.unique(vals, return_counts=True)

        if len(counts) > 0:
            return counts.mean(), sum(abs(counts - counts.mean())) / len(counts), len(counts)
        else:
            return None, None, 0

    def bar_plot_thing(
        self, x=None, thing=None, ax=None, mask=None, store=False, faults=False, name=None, op=None, args=None
    ):  # pragma: no cover
        '''
        Make a bar plot about something!

        Args:
            x (Numpy.ndarray of dimension 1): x values for bar plot
            thing (str): Some key stored in the stats that will go on the y-axis
            mask (Numpy.ndarray of shape (n)): The mask you want to apply before plotting
            store (bool): Store the plot at a predefined path or not (for jupyter notebooks)
            faults (bool): Whether to load stats with faults or without
            name (str): Optional name for the plot
            op (function): Operation that is applied to thing before plotting default is recovery rate
            args (dict): Parameters for how the plot should look

        Returns:
            None
        '''
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            store = True
        op = self.mean if op is None else op

        # get the values for the bars
        height = np.zeros(len(self.strategies))
        for strategy_idx in range(len(self.strategies)):
            strategy = self.strategies[strategy_idx]

            # load the values
            dat = self.load(strategy=strategy, faults=faults)
            no_faults = self.load(strategy=strategy, faults=False)

            # check if we have a mask
            if mask is None:
                mask = np.ones_like(dat[thing], dtype=bool)

            height[strategy_idx] = op(dat, no_faults, thing, mask)

        # get some x values
        x = np.arange(len(self.strategies)) if x is None else x

        # prepare labels
        ticks = [strategy.bar_plot_x_label for strategy in self.strategies]

        ax.bar(x, height, tick_label=ticks)

        # set the parameters
        ax.set_ylabel(thing)
        args = {} if args is None else args
        [plt.setp(ax, k, v) for k, v in args.items()]

        if store:
            fig.tight_layout()
            plt.savefig(f'data/{self.get_name()}-{thing if name is None else name}-barplot.pdf', transparent=True)
            plt.close(fig)

        return None

    def fault_frequency_plot(self, ax, iter_ax, kwargs_range, strategy=None):  # pragma: no cover
        func_args = locals()
        func_args.pop('self', None)
        if strategy is None:
            for strat in self.strategies:
                args = {**func_args, 'strategy': strat}
                self.fault_frequency_plot(**args)
            return None

        # load data
        all_data = {}
        for me in kwargs_range['fault_frequency_iter']:
            self.kwargs['fault_frequency_iter'] = me
            self.get_recovered()
            all_data[me] = self.load(strategy=strategy, faults=True, mode='regular')

        # get_recovery_rate
        results = {}
        results['frequencies'] = list(all_data.keys())
        results['recovery_rate'] = [
            len(all_data[key]['recovered'][all_data[key]['recovered'] is True]) / len(all_data[key]['recovered'])
            for key in all_data.keys()
        ]
        # results['iterations'] = [np.mean(all_data[key]['total_iteration']) for key in all_data.keys()]
        results['iterations'] = [
            np.mean(all_data[key]['total_iteration'][all_data[key]['error'] != np.inf]) for key in all_data.keys()
        ]

        ax.plot(results['frequencies'], results['recovery_rate'], **strategy.style)
        iter_ax.plot(results['frequencies'], results['iterations'], **{**strategy.style, 'ls': '--'})

        ax.set_xscale('log')
        ax.set_xlabel('iterations between fault')
        ax.set_ylabel('recovery rate')
        iter_ax.set_ylabel('average total iterations if not crashed (dashed)')
        ax.legend(frameon=False)

        return None


def check_local_error():  # pragma: no cover
    """
    Make a plot of the resolution over time for all problems
    """
    problems = [run_vdp, run_Lorenz, run_Schroedinger, run_quench]
    problems = [run_quench]
    strategies = [BaseStrategy(), AdaptivityStrategy(), IterateStrategy()]

    for i in range(len(problems)):
        stats_analyser = FaultStats(
            prob=problems[i],
            strategies=strategies,
            faults=[False],
            reload=True,
            recovery_thresh=1.1,
            num_procs=1,
            mode='random',
        )
        stats_analyser.compare_strategies()
    plt.show()


def main():
    stats_analyser = FaultStats(
        prob=run_vdp,
        strategies=[BaseStrategy(), AdaptivityStrategy(), IterateStrategy(), HotRodStrategy()],
        faults=[False, True],
        reload=True,
        recovery_thresh=1.1,
        # recovery_thresh_abs=1e-5,
        num_procs=1,
        mode='random',
        stats_path='data/stats-jusuf',
    )
    ########################
    # msk = stats_analyser.get_mask(AdaptivityStrategy(), val=False, key='recovered')
    # stats_analyser.print_faults(msk)
    fig, ax = plt.subplots()
    iter_ax = ax.twinx()
    kwargs_range = {'fault_frequency_iter': (10, 100, 1000, 10000)}
    stats_analyser.run_stats_generation(runs=10, kwargs_range=kwargs_range)
    stats_analyser.fault_frequency_plot(ax=ax, iter_ax=iter_ax, kwargs_range=kwargs_range)
    # stats_analyser.scrutinize(AdaptivityStrategy(), 4, True)
    plt.show()
    return None
    ########################

    stats_analyser.run_stats_generation(runs=5000)

    if MPI.COMM_WORLD.rank > 0:  # make sure only one rank accesses the data
        return None

    stats_analyser.get_recovered()
    mask = None

    # stats_analyser.compare_strategies()
    stats_analyser.plot_things_per_things(
        'recovered', 'node', False, op=stats_analyser.rec_rate, mask=mask, args={'ylabel': 'recovery rate'}
    )
    stats_analyser.plot_things_per_things(
        'recovered', 'iteration', False, op=stats_analyser.rec_rate, mask=mask, args={'ylabel': 'recovery rate'}
    )

    stats_analyser.plot_things_per_things(
        'recovered', 'bit', False, op=stats_analyser.rec_rate, mask=mask, args={'ylabel': 'recovery rate'}
    )

    # make a plot for only the faults that can be recovered
    fig, ax = plt.subplots(1, 1)
    for strategy in stats_analyser.strategies:
        fixable = stats_analyser.get_fixable_faults_only(strategy=strategy)
        stats_analyser.plot_things_per_things(
            'recovered',
            'bit',
            False,
            strategies=[strategy],
            op=stats_analyser.rec_rate,
            mask=fixable,
            args={'ylabel': 'recovery rate'},
            name='fixable_recovery',
            ax=ax,
        )
    fig.tight_layout()
    fig.savefig(f'data/{stats_analyser.get_name()}-recoverable.pdf', transparent=True)

    fig, ax = plt.subplots(1, 1, figsize=(13, 4))
    stats_analyser.plot_recovery_thresholds(ax=ax, thresh_range=np.logspace(-1, 1, 1000))
    ax.axvline(stats_analyser.get_thresh(BaseStrategy()), color='grey', ls=':', label='recovery threshold')
    ax.set_xscale('log')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f'data/{stats_analyser.get_name()}-threshold.pdf', transparent=True)

    stats_analyser.plot_things_per_things(
        'total_iteration',
        'bit',
        True,
        op=stats_analyser.mean,
        mask=mask,
        args={'yscale': 'log', 'ylabel': 'total iterations'},
    )
    stats_analyser.plot_things_per_things(
        'total_iteration',
        'bit',
        True,
        op=stats_analyser.extra_mean,
        mask=mask,
        args={'yscale': 'linear', 'ylabel': 'extra iterations'},
        name='extra_iter',
    )
    stats_analyser.plot_things_per_things(
        'error', 'bit', True, op=stats_analyser.mean, mask=mask, args={'yscale': 'log'}
    )

    stats_analyser.plot_recovery_thresholds()


if __name__ == "__main__":
    # check_local_error()
    main()
