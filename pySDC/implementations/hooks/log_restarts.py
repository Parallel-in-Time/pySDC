from pySDC.core.Hooks import hooks


class LogRestarts(hooks):
    """
    Record restarts as `restart` at the beginning of the step.
    """

    def post_step(self, step, level_number):
        """
        Record here if the step was restarted.

        Args:
            step (pySDC.Step.step): Current step
            level_number (int): Current level
        """
        super().post_step(step, level_number)

        L = step.levels[level_number]

        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='restart',
            value=int(step.status.get('restart')),
        )
