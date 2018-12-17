from pySDC.core.Hooks import hooks


class stop_at_error_hook(hooks):

    def post_sweep(self, step, level_number):
        """
        Overwrite standard post iteration hook

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(stop_at_error_hook, self).post_sweep(step, level_number)

        # some abbreviations
        L = step.levels[0]
        P = L.prob

        L.sweep.compute_end_point()

        uex = P.u_exact(L.time + L.dt)
        # print(abs(uex - L.uend))

        if abs(uex - L.uend) < 1E-02:
            print('Stop iterating at %s' % step.status.iter)
            step.status.force_done = True

        return None
