from pySDC.core.Hooks import hooks


class LogWork(hooks):
    """
    Log the increment of all work counters in the problem between steps
    """

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
            self.__work_last_step = [
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
        for key in self.__work_last_step[level_number].keys():
            self.add_to_stats(
                process=step.status.slot,
                time=L.time + L.dt,
                level=L.level_index,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type=f'work_{key}',
                value=L.prob.work_counters[key].niter - self.__work_last_step[level_number][key],
            )
