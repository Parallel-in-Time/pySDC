from pySDC.core.Hooks import hooks


class LogCacheInvalidates(hooks):
    """
    Store the compression ratio at the end of each step as "u".
    """

    def post_step(self, step, level_number):
        """
        Record compression ratio at the end of the step

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number

        Returns:
            None
        """
        super().post_step(step, level_number)

        L = step.levels[level_number]

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="cache_invalidates",
            value=L.u[0].manager.cacheManager.cacheInvalidates,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="cache_hits",
            value=L.u[0].manager.cacheManager.cache_hits,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="cache_misses",
            value=L.u[0].manager.cacheManager.cache_misses,
        )
