from pySDC.core.Hooks import hooks


class LogTimeCompression(hooks):
    """
    Store the solution at the end of each step as "u".
    """

    def post_step(self, step, level_number):
        """
        Record solution at the end of the step

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
            type="execution_time_comp_nocache",
            value=L.u[0].manager.compression_time_nocache,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="execution_time_comp",
            value=L.u[0].manager.compression_time,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="execution_time_comp_update",
            value=L.u[0].manager.compression_time_update,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="comp_time_eviction",
            value=L.u[0].manager.compression_time_eviction,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="comp_time_put",
            value=L.u[0].manager.compression_time_put_only,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="total_time_comp",
            value=L.u[0].manager.compression_time_update + L.u[0].manager.compression_time,
        )


class LogTimeDecompression(hooks):
    """
    Store the solution at the end of each step as "u".
    """

    def post_step(self, step, level_number):
        """
        Record solution at the end of the step

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
            type="execution_time_decomp_nocache",
            value=L.u[0].manager.decompression_time_nocache,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="execution_time_decomp",
            value=L.u[0].manager.decompression_time,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="decomp_time_eviction",
            value=L.u[0].manager.decompression_time_eviction,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="decomp_time_get",
            value=L.u[0].manager.decompression_time_get,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="decomp_time_put",
            value=L.u[0].manager.decompression_time_put_only,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="total_time_decomp",
            value=L.u[0].manager.decompression_time_get + L.u[0].manager.decompression_time,
        )
