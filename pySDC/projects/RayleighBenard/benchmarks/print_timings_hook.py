from pySDC.core.hooks import Hooks

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
import time


class PrintGPUTimings(Hooks):
    def pre_step(self, step, level_number):
        """
        Hook called before each step

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().pre_step(step, level_number)
        self.__t0_step = self._get_event()

    def post_step(self, step, level_number):
        """
        Default routine called after each step or block

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().post_step(step, level_number)
        self.__t1_step = self._get_event()
        self.logger.log(
            level=50, msg=f'GPU timing step: {self._compute_time_elapsed(self.__t1_step, self.__t0_step):.8f}s'
        )

    def _compute_time_elapsed(self, event_after, event_before):
        event_after.synchronize()
        return cp.cuda.get_elapsed_time(event_before, event_after) / 1e3

    def _get_event(self):
        event = cp.cuda.Event()
        event.record()
        return event


class PrintCPUTimings(Hooks):
    def pre_step(self, step, level_number):
        """
        Hook called before each step

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().pre_step(step, level_number)
        self.__t0_step = self._get_event()

    def post_step(self, step, level_number):
        """
        Default routine called after each step or block

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().post_step(step, level_number)
        self.__t1_step = self._get_event()
        self.logger.log(
            level=50, msg=f'CPU timing step: {self._compute_time_elapsed(self.__t1_step, self.__t0_step):.8f}s'
        )

    def _compute_time_elapsed(self, event_after, event_before):
        return event_after - event_before

    def _get_event(self):
        return time.perf_counter()
