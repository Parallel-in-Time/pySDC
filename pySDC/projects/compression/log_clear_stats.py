from pySDC.core.Hooks import hooks


class LogClearStats(hooks):
    """
    Store the solution at the end of each step as "u".
    """

    def post_step(self, step, level_number):
        """
        Clears the stats at the end of each step that need to be cleared

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number

        Returns:
            None
        """
        super().post_step(step, level_number)
        L = step.levels[level_number]

        # Clear the variables and metrics required:
        L.u[0].manager.num_active_registered_var = 0
        L.u[0].manager.registerVars_time = 0
        L.u[0].manager.compression_time_nocache = 0
        L.u[0].manager.decompression_time_nocache = 0
        L.u[0].manager.compression_time = 0
        L.u[0].manager.decompression_time = 0
        L.u[0].manager.compression_time_update = 0
        L.u[0].manager.compression_time_eviction = 0
        L.u[0].manager.compression_time_put_only = 0
        L.u[0].manager.decompression_time_get = 0
        L.u[0].manager.decompression_time_eviction = 0
        L.u[0].manager.decompression_time_put_only = 0
        L.u[0].manager.num_compression_calls = 0
        L.u[0].manager.num_decompression_calls = 0
        L.u[0].manager.cacheHist = []
        L.u[0].manager.cacheManager.cacheInvalidates = 0
        L.u[0].manager.cacheManager.cache_hits = 0
        L.u[0].manager.cacheManager.cache_misses = 0
