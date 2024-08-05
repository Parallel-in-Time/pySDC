from pySDC.core.hooks import Hooks


class LogResidualComponentsPostIter(Hooks):
    """
    Logs the residual of all components of the system after each iteration.

    Parameters
    ----------
    Hooks : pySDC.core.hooks.Hooks
        Hook base class.
    """

    def post_iteration(self, step, level_number):
        r"""
        Default routine called after each iteration.

        Parameters
        ----------
        step : pySDC.core.step.Step
            Current step.
        level_number : pySDC.core.level.Level
            Current level number.
        """

        super().post_iteration(step, level_number)

        L = step.levels[level_number]
        P = L.prob

        L.sweep.compute_end_point()

        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='residual_comp_post_iter',
            value=L.sweep.residual_components,
        )
