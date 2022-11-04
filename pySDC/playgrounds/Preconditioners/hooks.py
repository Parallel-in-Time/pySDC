from pySDC.core.Hooks import hooks


class log_error_at_iterations(hooks):
    """
    store the solution at all iterations as well as the exact solution to the step to compute the error
    """

    def pre_step(self, step, level_number):
        """
        Record los conditiones initiales
        """
        super(log_error_at_iterations, self).pre_step(step, level_number)

        L = step.levels[level_number]
        u_exact = L.prob.u_exact(t=L.time + L.dt)

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='u',
            value=L.u[0],
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='u_exact',
            value=u_exact,
        )

    def post_iteration(self, step, level_number):
        """
        Record final solutions as well as step size and error estimates
        """
        super(log_error_at_iterations, self).post_iteration(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='u',
            value=L.uend,
        )


class log_cost(hooks):
    '''
    This class stores all relevant information for the cost function
    '''

    def post_step(self, step, level_number):

        super(log_cost, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        self.increment_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='u_old',
            value=L.uold[-1],
        )
        self.increment_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='u',
            value=L.u[-1],
        )
        self.increment_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='e_em',
            value=L.status.__dict__.get("error_embedded_estimate", None),
        )
        self.increment_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='k',
            value=step.status.iter,
        )
        self.increment_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='restarts',
            value=int(step.status.restart),
        )
