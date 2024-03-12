from pySDC.core.Hooks import hooks


class pde_hook(hooks):
    def __init__(self):
        super(pde_hook, self).__init__()

    def pre_run(self, step, level_number):
        """
        Overwrite default routine called before time-loop starts
        Args:
            step: the current step
            level_number: the current level number
        """
        super(pde_hook, self).pre_run(step, level_number)

        L = step.levels[level_number]
        P = L.prob
        if level_number == 0 and L.time == P.t0:
            P.write_solution(L.u[0], P.t0)

    def post_step(self, step, level_number):
        super(pde_hook, self).post_step(step, level_number)

        if level_number == 0:
            L = step.levels[level_number]
            P = L.prob
            P.write_solution(L.uend, L.time + L.dt)
