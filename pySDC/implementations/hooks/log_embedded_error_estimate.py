from pySDC.core.Hooks import hooks


class LogEmbeddedErrorEstimate(hooks):
    """
    Store the embedded error estimate at the end of each step as "error_embedded_estimate".
    """

    def log_error(self, step, level_number, appendix=''):
        L = step.levels[level_number]

        for flavour in ['', '_collocation']:
            if L.status.get(f'error_embedded_estimate{flavour}'):
                if flavour == '_collocation':
                    iter, value = L.status.error_embedded_estimate_collocation
                else:
                    iter = step.status.iter
                    value = L.status.error_embedded_estimate
                self.add_to_stats(
                    process=step.status.slot,
                    time=L.time + L.dt,
                    level=L.level_index,
                    iter=iter,
                    sweep=L.status.sweep,
                    type=f'error_embedded_estimate{flavour}{appendix}',
                    value=value,
                )

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
        self.log_error(step, level_number, appendix)


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
        super().post_iteration(step, level_number)
        self.log_error(step, level_number, '_post_iteration')

    def post_step(self, step, level_number):
        super().post_step(step, level_number, appendix='_post_iteration')
