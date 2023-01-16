from pySDC.core.Hooks import hooks


class LogGlobalError(hooks):
    """
    Log the global error with respect to `u_exact` defined in the problem class as "e_global".
    Be aware that this requires the problems to be compatible with this. We need some kind of "exact" solution for this
    to work, be it a reference solution or something analytical.
    """

    def post_step(self, step, level_number):

        super(LogGlobalError, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='e_global',
            value=abs(L.prob.u_exact(t=L.time + L.dt) - L.u[-1]),
        )


class LogLocalError(hooks):
    """
    Log the local error with respect to `u_exact` defined in the problem class as "e_local".
    Be aware that this requires the problems to be compatible with this. In particular, a reference solution needs to
    be made available from the initial conditions of the step, not of the run. Otherwise you compute the global error.
    """

    def post_step(self, step, level_number):

        super(LogLocalError, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='e_local',
            value=abs(L.prob.u_exact(t=L.time + L.dt, u_init=L.u[0], t_init=L.time) - L.u[-1]),
        )
