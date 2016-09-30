from pySDC.Level import level
import logging
import time

from pySDC.Stats import stats


class hooks(object):

    __slots__ = ('__level','t0')

    def __init__(self):
        """
        Initialization routine
        """
        self.__level = None
        self.t0 = None
        pass


    def __set_level(self,L):
        """
        Sets a reference to the current level (done in the initialization of the level)

        Args:
            L: current level
        """
        assert isinstance(L,level)
        self.__level = L


    @property
    def level(self):
        """
        Getter for the current level
        Returns:
            level
        """
        return self.__level


    def pre_step(self,status):
        """
        Hook called before each step
        """
        self.t0 = time.time()
        pass


    def dump_pre(self,status):
        """
        Default routine called before time-loop starts
        """
        pass

    def dump_pre_iteration(self,status):
        """
        Default routine called before iteration starts
        """
        pass


    def dump_sweep(self,status):
        """
        Default routine called after each sweep
        """
        L = self.level
        logger = logging.getLogger('root')
        logger.info('Process %2i on time %8.6f at stage %15s: Level: %s -- Iteration: %2i -- Residual: %12.8e',
                    status.slot,L.time,status.stage,L.id,status.iter,L.status.residual)

        stats.add_to_stats(step=status.slot, time=L.time, level=L.id, iter=status.iter,
                           type='residual',  value=L.status.residual)

        pass


    def dump_iteration(self,status):
        """
        Default routine called after each iteration
        """
        L = self.level
        stats.add_to_stats(step=status.slot, time=L.time, iter=status.iter, type='residual',
                           value=L.status.residual)
        pass


    def dump_step(self,status):
        """
        Default routine called after each step
        """

        L = self.level
        stats.add_to_stats(step=status.slot, time=L.time, type='timing_step', value=time.time()-self.t0)
        stats.add_to_stats(step=status.slot, time=L.time, type='niter', value=status.iter)
        stats.add_to_stats(step=status.slot, time=L.time, type='residual', value=L.status.residual)

        pass