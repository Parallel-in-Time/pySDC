from pySDC.Level import level
import logging

from pySDC.Stats import stats


class hooks():

    __slots__ = ('__level')

    def __init__(self):
        """
        Initialization routine
        """
        self.__level = None
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

    def dump_pre(self,status):
        """
        Default routine called before time-loop starts
        """
        pass


    def dump_sweep(self,status):
        """
        Default routine called after each sweep
        """
        L = self.level
        logger = logging.getLogger('root')
        logger.info('Process %2i at stage %15s: Level: %s -- Iteration: %2i -- Residual: %12.8e',
                    status.slot,status.stage,L.id,status.iter,L.status.residual)

        stats.add_to_stats(time=status.time, level=L.id, iter=status.iter, type='residual', value=L.status.residual)

        pass


    def dump_iteration(self,status):
        """
        Default routine called after each iteration
        """
        L = self.level
        stats.add_to_stats(time=status.time, iter=status.iter, type='residual', value=L.status.residual)
        pass


    def dump_step(self,status):
        """
        Default routine called after each step
        """
        L = self.level
        stats.add_to_stats(time=status.time, type='niter', value=status.iter)
        stats.add_to_stats(time=status.time, type='residual', value=L.status.residual)

        pass