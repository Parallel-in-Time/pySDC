from pySDC.core.hooks import Hooks


class LogWork(Hooks):
    """
    Log the increment of all work counters in the problem between steps
    """

    def __init__(self):
        """
        Initialize the variables for the work recorded in the last step
        """
        super().__init__()
        self.__work_last_step = {}

    def pre_step(self, step, level_number):
        """
        Store the current values of the work counters

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number

        Returns:
            None
        """
        if level_number == 0:
            self.__work_last_step[step.status.slot] = [
                {key: step.levels[i].prob.work_counters[key].niter for key in step.levels[i].prob.work_counters.keys()}
                for i in range(len(step.levels))
            ]

    def post_step(self, step, level_number):
        """
        Add the difference between current values of counters and their values before the iteration to the stats.

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number

        Returns:
            None
        """
        L = step.levels[level_number]
        for key in self.__work_last_step[step.status.slot][level_number].keys():
            self.add_to_stats(
                process=step.status.slot,
                process_sweeper=L.sweep.rank,
                time=L.time + L.dt,
                level=L.level_index,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type=f'work_{key}',
                value=L.prob.work_counters[key].niter - self.__work_last_step[step.status.slot][level_number][key],
            )


class LogSDCIterations(Hooks):
    """
    Log the number of SDC iterations between steps.
    You can control the name in the stats via the class attribute ``name``.
    """

    name = 'k'

    def post_step(self, step, level_number):
        super().post_step(step, level_number)

        L = step.levels[level_number]
        self.increment_stats(
            process=step.status.slot,
            process_sweeper=L.sweep.rank,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type=self.name,
            value=step.status.iter,
        )
