import logging
import time
from collections import namedtuple


# noinspection PyUnusedLocal,PyShadowingBuiltins,PyShadowingNames
class hooks(object):
    """
    Hook class to contain the functions called during the controller runs (e.g. for calling user-routines)

    Attributes:
        __t0_setup (float): private variable to get starting time of setup
        __t0_run (float): private variable to get starting time of the run
        __t0_predict (float): private variable to get starting time of the predictor
        __t0_step (float): private variable to get starting time of the step
        __t0_iteration (float): private variable to get starting time of the iteration
        __t0_sweep (float): private variable to get starting time of the sweep
        __t0_comm (list): private variable to get starting time of the communication
        __t1_run (float): private variable to get end time of the run
        __t1_predict (float): private variable to get end time of the predictor
        __t1_step (float): private variable to get end time of the step
        __t1_iteration (float): private variable to get end time of the iteration
        __t1_sweep (float): private variable to get end time of the sweep
        __t1_setup (float): private variable to get end time of setup
        __t1_comm (list): private variable to hold timing of the communication (!)
        __num_restarts (int): number of restarts of the current step
        logger: logger instance for output
        __stats (dict): dictionary for gathering the statistics of a run
        __entry (namedtuple): statistics entry containing all information to identify the value
    """

    def __init__(self):
        """
        Initialization routine
        """
        self.__t0_setup = None
        self.__t0_run = None
        self.__t0_predict = None
        self.__t0_step = None
        self.__t0_iteration = None
        self.__t0_sweep = None
        self.__t0_comm = []
        self.__t1_run = None
        self.__t1_predict = None
        self.__t1_step = None
        self.__t1_iteration = None
        self.__t1_sweep = None
        self.__t1_setup = None
        self.__t1_comm = []
        self.__num_restarts = 0

        self.logger = logging.getLogger('hooks')

        # create statistics and entry elements
        self.__stats = {}
        self.__entry = namedtuple('Entry', ['process', 'time', 'level', 'iter', 'sweep', 'type', 'num_restarts'])

    def add_to_stats(self, process, time, level, iter, sweep, type, value):
        """
        Routine to add data to the statistics dict

        Args:
            process: the current process recording this data
            time (float): the current simulation time
            level (int): the current level index
            iter (int): the current iteration count
            sweep (int): the current sweep count
            type (str): string to describe the type of value
            value: the actual data
        """
        # create named tuple for the key and add to dict
        self.__stats[
            self.__entry(
                process=process,
                time=time,
                level=level,
                iter=iter,
                sweep=sweep,
                type=type,
                num_restarts=self.__num_restarts,
            )
        ] = value

    def increment_stats(self, process, time, level, iter, sweep, type, value, initialize=None):
        """
        Routine to increment data to the statistics dict. If the data is not yet created, it will be initialized to
        initialize if applicable and to value otherwise

        Args:
            process: the current process recording this data
            time (float): the current simulation time
            level (int): the current level index
            iter (int): the current iteration count
            sweep (int): the current sweep count
            type (str): string to describe the type of value
            value: the actual data
            initialize: if supplied and data does not exist already, this will be used over value
        """
        key = self.__entry(
            process=process, time=time, level=level, iter=iter, sweep=sweep, type=type, num_restarts=self.__num_restarts
        )
        if key in self.__stats.keys():
            self.__stats[key] += value
        elif initialize is not None:
            self.__stats[key] = initialize
        else:
            self.__stats[key] = value

    def return_stats(self):
        """
        Getter for the stats

        Returns:
            dict: stats
        """
        return self.__stats

    def reset_stats(self):
        """
        Function to reset the stats for multiple runs
        """
        self.__stats = {}

    def pre_setup(self, step, level_number):
        """
        Default routine called before setup starts

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__num_restarts = step.status.get('restarts_in_a_row') if step is not None else 0
        self.__t0_setup = time.perf_counter()

    def pre_run(self, step, level_number):
        """
        Default routine called before time-loop starts

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__num_restarts = step.status.get('restarts_in_a_row') if step is not None else 0
        self.__t0_run = time.perf_counter()

    def pre_predict(self, step, level_number):
        """
        Default routine called before predictor starts

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__t0_predict = time.perf_counter()

    def pre_step(self, step, level_number):
        """
        Hook called before each step

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__num_restarts = step.status.get('restarts_in_a_row') if step is not None else 0
        self.__t0_step = time.perf_counter()

    def pre_iteration(self, step, level_number):
        """
        Default routine called before iteration starts

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__num_restarts = step.status.get('restarts_in_a_row') if step is not None else 0
        self.__t0_iteration = time.perf_counter()

    def pre_sweep(self, step, level_number):
        """
        Default routine called before sweep starts

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__num_restarts = step.status.get('restarts_in_a_row') if step is not None else 0
        self.__t0_sweep = time.perf_counter()

    def pre_comm(self, step, level_number):
        """
        Default routine called before communication starts

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__num_restarts = step.status.get('restarts_in_a_row') if step is not None else 0
        if len(self.__t0_comm) >= level_number + 1:
            self.__t0_comm[level_number] = time.perf_counter()
        else:
            while len(self.__t0_comm) < level_number:
                self.__t0_comm.append(None)
            self.__t0_comm.append(time.perf_counter())
            while len(self.__t1_comm) <= level_number:
                self.__t1_comm.append(0.0)
            assert len(self.__t0_comm) == level_number + 1
            assert len(self.__t1_comm) == level_number + 1

    def post_comm(self, step, level_number, add_to_stats=False):
        """
        Default routine called after each communication

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
            add_to_stats (bool): set if result should go to stats object
        """
        self.__num_restarts = step.status.get('restarts_in_a_row') if step is not None else 0
        assert len(self.__t1_comm) >= level_number + 1
        self.__t1_comm[level_number] += time.perf_counter() - self.__t0_comm[level_number]

        if add_to_stats:
            L = step.levels[level_number]

            self.add_to_stats(
                process=step.status.slot,
                time=L.time,
                level=L.level_index,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type='timing_comm',
                value=self.__t1_comm[level_number],
            )
            self.__t1_comm[level_number] = 0.0

    def post_sweep(self, step, level_number):
        """
        Default routine called after each sweep

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__num_restarts = step.status.get('restarts_in_a_row') if step is not None else 0
        self.__t1_sweep = time.perf_counter()

        L = step.levels[level_number]

        self.logger.info(
            'Process %2i on time %8.6f at stage %15s: Level: %s -- Iteration: %2i -- Sweep: %2i -- ' 'residual: %12.8e',
            step.status.slot,
            L.time,
            step.status.stage,
            L.level_index,
            step.status.iter,
            L.status.sweep,
            L.status.residual,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='residual_post_sweep',
            value=L.status.residual,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='timing_sweep',
            value=self.__t1_sweep - self.__t0_sweep,
        )

    def post_iteration(self, step, level_number):
        """
        Default routine called after each iteration

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__num_restarts = step.status.get('restarts_in_a_row') if step is not None else 0

        self.__t1_iteration = time.perf_counter()

        L = step.levels[level_number]

        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=-1,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='residual_post_iteration',
            value=L.status.residual,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='timing_iteration',
            value=self.__t1_iteration - self.__t0_iteration,
        )

    def post_step(self, step, level_number):
        """
        Default routine called after each step or block

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__num_restarts = step.status.get('restarts_in_a_row') if step is not None else 0

        self.__t1_step = time.perf_counter()

        L = step.levels[level_number]

        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='timing_step',
            value=self.__t1_step - self.__t0_step,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=-1,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='niter',
            value=step.status.iter,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=-1,
            sweep=L.status.sweep,
            type='residual_post_step',
            value=L.status.residual,
        )

        # record the recomputed quantities at weird positions to make sure there is only one value for each step
        for t in [L.time, L.time + L.dt]:
            self.add_to_stats(
                process=-1, time=t, level=-1, iter=-1, sweep=-1, type='_recomputed', value=step.status.get('restart')
            )

    def post_predict(self, step, level_number):
        """
        Default routine called after each predictor

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__num_restarts = step.status.get('restarts_in_a_row') if step is not None else 0
        self.__t1_predict = time.perf_counter()

        L = step.levels[level_number]

        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='timing_predictor',
            value=self.__t1_predict - self.__t0_predict,
        )

    def post_run(self, step, level_number):
        """
        Default routine called after each run

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__num_restarts = step.status.get('restarts_in_a_row') if step is not None else 0
        self.__t1_run = time.perf_counter()

        L = step.levels[level_number]

        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='timing_run',
            value=self.__t1_run - self.__t0_run,
        )

    def post_setup(self, step, level_number):
        """
        Default routine called after setup

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__num_restarts = step.status.get('restarts_in_a_row') if step is not None else 0
        self.__t1_setup = time.perf_counter()

        self.add_to_stats(
            process=-1,
            time=-1,
            level=-1,
            iter=-1,
            sweep=-1,
            type='timing_setup',
            value=self.__t1_setup - self.__t0_setup,
        )
