from pySDC.core.Hooks import hooks


class get_update(hooks):

    def pre_iteration(self, step, level_number):
        """
        Hook called before each step

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(get_update, self).pre_iteration(step, level_number)

        L = step.levels[level_number]
        L.sweep.compute_end_point()

        L.uold = L.uend

    def post_iteration(self, step, level_number):
        """
        Hook called before each step

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(get_update, self).post_iteration(step, level_number)

        L = step.levels[level_number]
        L.sweep.compute_end_point()

        uex = L.prob.u_exact(L.time + L.dt)
        err_new = abs(uex - L.uend)

        print(abs(L.uold - L.uend), err_new)