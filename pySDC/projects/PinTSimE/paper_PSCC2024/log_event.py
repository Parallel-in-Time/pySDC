from pySDC.core.Hooks import hooks


class LogEventDiscontinuousTestDAE(hooks):
    """
    Logs the data for the discontinuous test DAE problem containing one discrete event.
    Note that this logging data is dependent from the problem itself.
    """

    def post_step(self, step, level_number):
        super(LogEventDiscontinuousTestDAE, self).post_step(step, level_number)

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
            value=2 * L.uend[0] - 100,
        )


class LogEventWSCC9(hooks):
    """
    Logs the data for the discontinuous test DAE problem containing one discrete event.
    Note that this logging data is dependent from the problem itself.
    """

    def post_step(self, step, level_number):
        super(LogEventWSCC9, self).post_step(step, level_number)

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
            value=L.uend[10 * P.m] - P.psv_max,
        )