import logging
import time
from collections import namedtuple

from pySDC.Plugins.pysdc_helper import FrozenClass


class hooks(FrozenClass):
    """
    Hook class to contain the functions called during the controller runs (e.g. for calling user-routines)

    Attributes:
        __t0: private variable to get starting time of the step
        logger: logger instance for output
        __stats: dictionary for gathering the statistics of a run
        __entry: statistics entry containign all information to identify the value
    """

    def __init__(self):
        """
        Initialization routine
        """
        self.__t0 = None
        self.logger = logging.getLogger('hooks')

        # create statistics and entry elements
        self.__stats = {}
        self.__entry = namedtuple('Entry', ['process', 'time', 'level', 'iter', 'type'])

        # don't allow adding more stuff to this class
        self._freeze()
        pass

    def add_to_stats(self,process,time,level,iter,type,value):
        """
        Routine to add data to the statistics dict

        Args:
            process: the current process recording this data
            time: the current simulation time
            level: the current level id
            iter: the current iteration count
            type: string to describe the type of value
            value: the actual data
        """
        # create named tuple for the key and add to dict
        self.__stats[self.__entry(process=process,time=time,level=level,iter=iter,type=type)] = value

    def return_stats(self):
        """
        Getter for the stats
        Returns:
            stats
        """
        return self.__stats

    def pre_step(self, step, level_number):
        """
        Hook called before each step
        Args:
            step: the current step
            level_number: the current level number
        """
        self.__t0 = time.time()
        pass


    def dump_pre(self, step, level_number):
        """
        Default routine called before time-loop starts
        Args:
            step: the current step
            level_number: the current level number
        """
        pass

    def dump_pre_iteration(self, step, level_number):
        """
        Default routine called before iteration starts
        Args:
            step: the current step
            level_number: the current level number
        """
        pass


    def dump_sweep(self, step, level_number):
        """
        Default routine called after each sweep
        Args:
            step: the current step
            level_number: the current level number
        """
        L = step.levels[level_number]

        self.logger.info('Process %2i on time %8.6f at stage %15s: Level: %s -- Iteration: %2i -- Residual: %12.8e',
                         step.status.slot,L.time,step.status.stage,L.id,step.status.iter,L.status.residual)

        self.add_to_stats(process=step.status.slot, time=L.time, level=L.id, iter=step.status.iter,
                           type='residual',  value=L.status.residual)

        pass


    def dump_iteration(self, step, level_number):
        """
        Default routine called after each iteration
        Args:
            step: the current step
            level_number: the current level number
        """
        L = step.levels[level_number]
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.id, iter=step.status.iter,
                           type='residual', value=L.status.residual)
        pass


    def dump_step(self, step, level_number):
        """
        Default routine called after each step
        Args:
            step: the current step
            level_number: the current level number
        """

        L = step.levels[level_number]
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.id, iter=step.status.iter,
                           type='timing_step', value=time.time()-self.__t0)
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.id, iter=step.status.iter,
                           type='niter', value=step.status.iter)

        pass