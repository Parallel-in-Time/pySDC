import logging
from collections import namedtuple


# noinspection PyUnusedLocal,PyShadowingBuiltins,PyShadowingNames
class hooks(object):
    """
    Hook class to contain the functions called during the controller runs (e.g. for calling user-routines)

    When deriving a custom hook from this class make sure to always call the parent method using e.g.
    `super().post_step(step, level_number)`. Otherwise bugs may arise when using `filer_recomputed` from the stats
    helper for post processing.

    Attributes:
        logger: logger instance for output
        __num_restarts (int): number of restarts of the current step
        __stats (dict): dictionary for gathering the statistics of a run
        __entry (namedtuple): statistics entry containing all information to identify the value
    """

    def __init__(self):
        """
        Initialization routine
        """
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

    def pre_run(self, step, level_number):
        """
        Default routine called before time-loop starts

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__num_restarts = step.status.get('restarts_in_a_row') if step is not None else 0

    def pre_predict(self, step, level_number):
        """
        Default routine called before predictor starts

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__num_restarts = step.status.get('restarts_in_a_row') if step is not None else 0

    def pre_step(self, step, level_number):
        """
        Hook called before each step

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__num_restarts = step.status.get('restarts_in_a_row') if step is not None else 0

    def pre_iteration(self, step, level_number):
        """
        Default routine called before iteration starts

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__num_restarts = step.status.get('restarts_in_a_row') if step is not None else 0

    def pre_sweep(self, step, level_number):
        """
        Default routine called before sweep starts

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__num_restarts = step.status.get('restarts_in_a_row') if step is not None else 0

    def pre_comm(self, step, level_number):
        """
        Default routine called before communication starts

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__num_restarts = step.status.get('restarts_in_a_row') if step is not None else 0

    def post_comm(self, step, level_number, add_to_stats=False):
        """
        Default routine called after each communication

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
            add_to_stats (bool): set if result should go to stats object
        """
        self.__num_restarts = step.status.get('restarts_in_a_row') if step is not None else 0

    def post_sweep(self, step, level_number):
        """
        Default routine called after each sweep

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__num_restarts = step.status.get('restarts_in_a_row') if step is not None else 0

    def post_iteration(self, step, level_number):
        """
        Default routine called after each iteration

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__num_restarts = step.status.get('restarts_in_a_row') if step is not None else 0

    def post_step(self, step, level_number):
        """
        Default routine called after each step or block

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__num_restarts = step.status.get('restarts_in_a_row') if step is not None else 0

    def post_predict(self, step, level_number):
        """
        Default routine called after each predictor

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__num_restarts = step.status.get('restarts_in_a_row') if step is not None else 0

    def post_run(self, step, level_number):
        """
        Default routine called after each run

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__num_restarts = step.status.get('restarts_in_a_row') if step is not None else 0

    def post_setup(self, step, level_number):
        """
        Default routine called after setup

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__num_restarts = step.status.get('restarts_in_a_row') if step is not None else 0
