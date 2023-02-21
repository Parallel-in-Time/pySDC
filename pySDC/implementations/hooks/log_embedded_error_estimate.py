from pySDC.core.Hooks import hooks


class LogEmbeddedErrorEstimate(hooks):
    """
    Store the embedded error estimate at the end of each step as "error_embedded_estimate".
    """

    def post_step(self, step, level_number, appendix=''):
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
            type=f'error_embedded_estimate{appendix}',
            value=L.status.get('error_embedded_estimate'),
        )


class LogEmbeddedErrorEstimatePostIter(LogEmbeddedErrorEstimate):
    """
    Store the embedded error estimate after each iteration as "error_embedded_estimate_post_iteration".

    Because the error estimate is computed after the hook is called, we record the value belonging to the last
    iteration, which is also why we need to record something after the step, which belongs to the final iteration.
    """

    def post_iteration(self, step, level_number):
        """
        Record embedded error estimate

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number

        Returns:
            None
        """
        # check if the estimate is available at all
        super().post_iteration(step, level_number)

        L = step.levels[level_number]

        if not L.status.get('error_embedded_estimate'):
            return None

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter - 1,
            sweep=L.status.sweep,
            type='error_embedded_estimate_post_iteration',
            value=L.status.get('error_embedded_estimate'),
        )

    def post_step(self, step, level_number):
        super().post_step(step, level_number, appendix='_post_iteration')
