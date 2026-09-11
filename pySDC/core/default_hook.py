import time
from pySDC.core.hooks import Hooks


class DefaultHooks(Hooks):
    """
    Hook class to contain the functions called during the controller runs (e.g. for calling user-routines)
    """

    def post_sweep(self, step, level_number):
        """
        Default routine called after each sweep

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().post_sweep(step, level_number)

        L = step.levels[level_number]

        self.logger.info(
            'Process %2i on time %8.6f at stage %15s: Level: %s -- Iteration: %2i -- Sweep: %2i -- ' 'residual: %12.8e',
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

    def post_iteration(self, step, level_number):
        """
        Default routine called after each iteration

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().post_iteration(step, level_number)

        L = step.levels[level_number]

        self.add_to_stats(
            process=step.status.slot,
            process_sweeper=L.sweep.rank,
            time=L.time,
            level=-1,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='residual_post_iteration',
            value=L.status.residual,
        )

    def post_step(self, step, level_number):
        """
        Default routine called after each step or block

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().post_step(step, level_number)

        L = step.levels[level_number]

        self.add_to_stats(
            process=step.status.slot,
            process_sweeper=L.sweep.rank,
            time=L.time,
            level=-1,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='niter',
            value=step.status.iter,
        )
        self.add_to_stats(
            process=step.status.slot,
            process_sweeper=L.sweep.rank,
            time=L.time,
            level=L.level_index,
            iter=-1,
            sweep=L.status.sweep,
            type='residual_post_step',
            value=L.status.residual,
        )

        # record the recomputed quantities at weird positions to make sure there is only one value for each step
        for t in [L.time, L.time + L.dt]:
            self.add_to_stats(
                process=-1,
                time=t,
                level=-1,
                iter=-1,
                sweep=-1,
                type='_recomputed',
                value=step.status.get('restart'),
                process_sweeper=-1,
            )
