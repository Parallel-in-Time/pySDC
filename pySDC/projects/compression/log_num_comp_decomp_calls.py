from pySDC.core.Hooks import hooks


class LogCompDecompCalls(hooks):
    """
    Store the number of function calls for compression and decompression at the end of each step as "u".
    """

    def post_step(self, step, level_number):
        """
        Record number of function calls at the end of the step

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
            type="num_comp_decomp_calls",
            value=L.u[0].manager.num_compression_calls + L.u[0].manager.num_decompression_calls,
        )


class LogCompCalls(hooks):
    """
    Store the number of function calls for compression at the end of each step as "u".
    """

    def post_step(self, step, level_number):
        """
        Record number of function calls at the end of the step

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
            type="num_comp_calls",
            value=L.u[0].manager.num_compression_calls,
        )


class LogDecompCalls(hooks):
    """
    Store the number of function calls for decompression at the end of each step as "u".
    """

    def post_step(self, step, level_number):
        """
        Record number of function calls at the end of the step

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
            type="num_decomp_calls",
            value=L.u[0].manager.num_decompression_calls,
        )
