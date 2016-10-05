from pySDC.Level import level
import logging
import time

from pySDC.Stats import stats


class hooks(object):

    __slots__ = ('__level','t0','logger')

    def __init__(self):
        """
        Initialization routine
        """
        self.t0 = None
        self.logger = logging.getLogger('hooks')
        pass

    def pre_step(self, step, level_number):
        """
        Hook called before each step
        Args:
            step: the current step
            level_number: the current level number
        """
        self.t0 = time.time()
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

        stats.add_to_stats(process=step.status.slot, time=L.time, level=L.id, iter=step.status.iter,
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
        stats.add_to_stats(process=step.status.slot, time=L.time, level=L.id, iter=step.status.iter,
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
        stats.add_to_stats(process=step.status.slot, time=L.time, level=L.id, iter=step.status.iter,
                           type='timing_step', value=time.time()-self.t0)
        stats.add_to_stats(process=step.status.slot, time=L.time, level=L.id, iter=step.status.iter,
                           type='niter', value=step.status.iter)

        pass