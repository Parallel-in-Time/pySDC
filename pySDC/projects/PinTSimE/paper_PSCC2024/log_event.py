from pySDC.core.hooks import Hooks


class LogEventDiscontinuousTestDAE(Hooks):
    """
    Logs the data for the discontinuous test DAE problem containing one discrete event.
    Note that this logging data is dependent from the problem itself.
    """

    def post_step(self, step, level_number):
        super().post_step(step, level_number)

        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='state_function',
            value=2 * L.uend.diff[0] - 100,
        )


class LogEventWSCC9(Hooks):
    """
    Logs the data for the discontinuous test DAE problem containing one discrete event.
    Note that this logging data is dependent from the problem itself.
    """

    def post_step(self, step, level_number):
        super().post_step(step, level_number)

        L = step.levels[level_number]
        P = L.prob

        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='state_function',
            value=L.uend.diff[10 * P.m] - P.psv_max,
        )
