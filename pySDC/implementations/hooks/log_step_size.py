from pySDC.core.Hooks import hooks


class LogStepSize(hooks):
    """
    Store the step size at the end of each step as "dt".
    """

    def post_step(self, step, level_number):
        """
        Record step size

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number

        Returns:
            None
        """
        # some abbreviations
        L = step.levels[level_number]

        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='dt',
            value=L.dt,
        )
