from pySDC.core.Hooks import hooks


class LogEmbeddedErrorEstimate(hooks):
    """
    Store the embedded error estimate at the end of each step as "error_embedded_estimate".
    """

    def post_step(self, step, level_number):
        """
        Record embedded error estimate

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number

        Returns:
            None
        """
        super().post_step(step, level_number)

        L = step.levels[level_number]

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='error_embedded_estimate',
            value=L.status.get('error_embedded_estimate'),
        )
