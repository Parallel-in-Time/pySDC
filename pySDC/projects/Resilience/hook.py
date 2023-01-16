from pySDC.core.Hooks import hooks
from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.implementations.hooks.log_embedded_error_estimate import LogEmbeddedErrorEstimate
from pySDC.implementations.hooks.log_extrapolated_error_estimate import LogExtrapolationErrorEstimate
from pySDC.implementations.hooks.log_step_size import LogStepSize


hook_collection = [LogSolution, LogEmbeddedErrorEstimate, LogExtrapolationErrorEstimate, LogStepSize]


class log_data(hooks):
    """
    Record data required for analysis of problems in the resilience project
    """

    def pre_run(self, step, level_number):
        """
        Record initial conditions
        """
        super().pre_run(step, level_number)

        L = step.levels[level_number]
        self.add_to_stats(process=0, time=0, level=0, iter=0, sweep=0, type='u0', value=L.u[0])

    def post_step(self, step, level_number):
        """
        Record final solutions as well as step size and error estimates
        """
        super().post_step(step, level_number)

        L = step.levels[level_number]

        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='restart',
            value=int(step.status.get('restart')),
        )
        # add the following with two names because I use both in different projects -.-
        for k in ['sweeps', 'k']:
            self.increment_stats(
                process=step.status.slot,
                time=L.time,
                level=L.level_index,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type=k,
                value=step.status.iter,
            )


class LogUAllIter(hooks):
    """
    Log solution and errors after each iteration
    """

    def post_iteration(self, step, level_number):
        super(LogUAllIter, self).post_iteration(step, level_number)
        # some abbreviations
        L = step.levels[level_number]
        iter = step.status.iter

        L.sweep.compute_end_point()
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=iter,
            sweep=L.status.sweep,
            type='u',
            value=L.uend,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=iter,
            sweep=L.status.sweep,
            type='e_em',
            value=L.status.get('error_embedded_estimate'),
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=iter,
            sweep=L.status.sweep,
            type='e_ex',
            value=L.status.get('error_extrapolation_estimate'),
        )
