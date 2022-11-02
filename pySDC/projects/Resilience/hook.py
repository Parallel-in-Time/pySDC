from pySDC.core.Hooks import hooks

class log_error_estimates(hooks):
    """
    Record data required for analysis of problems in the resilience project
    """
    def pre_run(self, step, level_number):
        """
        Record los conditiones initiales
        """
        super(log_error_estimates, self).pre_run(step, level_number)
        L = step.levels[level_number]
        self.add_to_stats(process=0, time=0, level=0, iter=0, sweep=0, type='u0', value=L.u[0])

    def post_step(self, step, level_number):
        """
        Record final solutions as well as step size and error estimates
        """
        super(log_error_estimates, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='u',
            value=L.uend,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='dt',
            value=L.dt,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='e_embedded',
            value=L.status.__dict__.get('error_embedded_estimate', None),
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='e_extrapolated',
            value=L.status.__dict__.get('error_extrapolation_estimate', None),
        )
