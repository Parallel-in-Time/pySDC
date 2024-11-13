import time
from pySDC.core.hooks import Hooks

try:
    import cupy as cp
except ImportError:
    cp = None


class Timings(Hooks):
    """
    Abstract base class for recoding timings

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
    """

    prefix = ''

    def _compute_time_elapsed(self, event_after, event_before):
        raise NotImplementedError

    def _get_event(self):
        raise NotImplementedError

    def __init__(self):
        super().__init__()
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

    def pre_setup(self, step, level_number):
        """
        Default routine called before setup starts

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().pre_setup(step, level_number)
        self.__t0_setup = self._get_event()

    def pre_run(self, step, level_number):
        """
        Default routine called before time-loop starts

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().pre_run(step, level_number)
        self.__t0_run = self._get_event()

    def pre_predict(self, step, level_number):
        """
        Default routine called before predictor starts

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().pre_predict(step, level_number)
        self.__t0_predict = self._get_event()

    def pre_step(self, step, level_number):
        """
        Hook called before each step

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().pre_step(step, level_number)
        self.__t0_step = self._get_event()

    def pre_iteration(self, step, level_number):
        """
        Default routine called before iteration starts

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().pre_iteration(step, level_number)
        self.__t0_iteration = self._get_event()

    def pre_sweep(self, step, level_number):
        """
        Default routine called before sweep starts

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().pre_sweep(step, level_number)
        self.__t0_sweep = self._get_event()

    def pre_comm(self, step, level_number):
        """
        Default routine called before communication starts

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().pre_comm(step, level_number)
        if len(self.__t0_comm) >= level_number + 1:
            self.__t0_comm[level_number] = self._get_event()
        else:
            while len(self.__t0_comm) < level_number:
                self.__t0_comm.append(None)
            self.__t0_comm.append(self._get_event())
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
        super().post_comm(step, level_number)
        assert len(self.__t1_comm) >= level_number + 1
        self.__t1_comm[level_number] += self._compute_time_elapsed(self._get_event(), self.__t0_comm[level_number])

        if add_to_stats:
            L = step.levels[level_number]

            self.add_to_stats(
                process=step.status.slot,
                process_sweeper=L.sweep.rank,
                time=L.time,
                level=L.level_index,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type=f'{self.prefix}timing_comm',
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
        super().post_sweep(step, level_number)
        self.__t1_sweep = self._get_event()

        L = step.levels[level_number]

        self.add_to_stats(
            process=step.status.slot,
            process_sweeper=L.sweep.rank,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type=f'{self.prefix}timing_sweep',
            value=self._compute_time_elapsed(self.__t1_sweep, self.__t0_sweep),
        )

    def post_iteration(self, step, level_number):
        """
        Default routine called after each iteration

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().post_iteration(step, level_number)
        self.__t1_iteration = self._get_event()

        L = step.levels[level_number]

        self.add_to_stats(
            process=step.status.slot,
            process_sweeper=L.sweep.rank,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type=f'{self.prefix}timing_iteration',
            value=self._compute_time_elapsed(self.__t1_iteration, self.__t0_iteration),
        )

    def post_step(self, step, level_number):
        """
        Default routine called after each step or block

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().post_step(step, level_number)
        self.__t1_step = self._get_event()

        L = step.levels[level_number]

        self.add_to_stats(
            process=step.status.slot,
            process_sweeper=L.sweep.rank,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type=f'{self.prefix}timing_step',
            value=self._compute_time_elapsed(self.__t1_step, self.__t0_step),
        )

    def post_predict(self, step, level_number):
        """
        Default routine called after each predictor

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().post_predict(step, level_number)
        self.__t1_predict = self._get_event()

        L = step.levels[level_number]

        self.add_to_stats(
            process=step.status.slot,
            process_sweeper=L.sweep.rank,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type=f'{self.prefix}timing_predictor',
            value=self._compute_time_elapsed(self.__t1_predict, self.__t0_predict),
        )

    def post_run(self, step, level_number):
        """
        Default routine called after each run

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().post_run(step, level_number)
        self.__t1_run = self._get_event()
        t_run = self._compute_time_elapsed(self.__t1_run, self.__t0_run)

        L = step.levels[level_number]

        self.add_to_stats(
            process=step.status.slot,
            process_sweeper=L.sweep.rank,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type=f'{self.prefix}timing_run',
            value=t_run,
        )
        self.logger.info(f'Finished run after {t_run:.2e}s')

    def post_setup(self, step, level_number):
        """
        Default routine called after setup

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().post_setup(step, level_number)
        self.__t1_setup = self._get_event()

        self.add_to_stats(
            process=-1,
            process_sweeper=-1,
            time=-1,
            level=-1,
            iter=-1,
            sweep=-1,
            type=f'{self.prefix}timing_setup',
            value=self._compute_time_elapsed(self.__t1_setup, self.__t0_setup),
        )


class CPUTimings(Timings):
    """
    Hook for recording CPU timings of important operations during a pySDC run.
    """

    def _compute_time_elapsed(self, event_after, event_before):
        return event_after - event_before

    def _get_event(self):
        return time.perf_counter()


class GPUTimings(Timings):
    """
    Hook for recording GPU timings of important operations during a pySDC run.
    """

    prefix = 'GPU_'

    def _compute_time_elapsed(self, event_after, event_before):
        event_after.synchronize()
        return cp.cuda.get_elapsed_time(event_before, event_after) / 1e3

    def _get_event(self):
        event = cp.cuda.Event()
        event.record()
        return event
