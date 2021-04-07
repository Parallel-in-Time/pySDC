from pySDC.core.Hooks import hooks


class error_output(hooks):
    """
    Hook class to add output of error
    """

    def post_step(self, step, level_number):
        """
        Default routine called after each step
        Args:
            step: the current step
            level_number: the current level number
        """

        super(error_output, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]
        P = L.prob

        L.sweep.compute_end_point()

        # compute and save errors
        upde = P.u_exact(step.time + step.dt)
        pde_err = abs(upde - L.uend)

        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=step.status.iter,
                          sweep=L.status.sweep, type='error_after_step', value=pde_err)
        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=step.status.iter,
                          sweep=L.status.sweep, type='residual_after_step', value=L.status.residual)
