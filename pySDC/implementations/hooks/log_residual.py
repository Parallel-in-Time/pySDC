from pySDC.core.hooks import Hooks


class LogResidualPostSweep(Hooks):
    """
    Compute the residual after every sweep, on every level, and store it as "residual_post_sweep".

    The controllers compute the residual only once per iteration, where the convergence check needs it: after the new
    initial value has come in. This hook adds the residual after each sweep for those who want to watch the levels
    converge, at the cost of one residual per sweep. It is computed with the sweep's stage, so stages listed in the
    sweeper's `skip_residual_computation` are still skipped.
    """

    def post_sweep(self, step, level_number):
        """
        Compute and record the residual

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number

        Returns:
            None
        """
        super().post_sweep(step, level_number)

        L = step.levels[level_number]
        L.sweep.compute_residual(stage=step.status.stage)

        self.logger.info(
            'Process %2i on time %8.6f at stage %15s: Level: %s -- Iteration: %2i -- Sweep: %2i -- residual: %12.8e',
            step.status.slot,
            L.time,
            step.status.stage,
            L.level_index,
            step.status.iter,
            L.status.sweep,
            L.status.residual,
        )

        self.add_to_stats(
            process=step.status.slot,
            process_sweeper=L.sweep.rank,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='residual_post_sweep',
            value=L.status.residual,
        )
