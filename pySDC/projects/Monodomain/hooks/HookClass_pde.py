from pySDC.core.hooks import Hooks


class pde_hook(Hooks):
    """
    Hook class to write the solution to file.
    """

    def __init__(self):
        super(pde_hook, self).__init__()

    def pre_run(self, step, level_number):
        """
        Overwrite default routine called before time-loop starts
        It calls the default routine and then writes the initial value to file.
        """
        super(pde_hook, self).pre_run(step, level_number)

        L = step.levels[level_number]
        P = L.prob
        if level_number == 0 and L.time == P.t0:
            P.write_solution(L.u[0], P.t0)

    def post_step(self, step, level_number):
        """
        Overwrite default routine called after each step.
        It calls the default routine and then writes the solution to file.
        """
        super(pde_hook, self).post_step(step, level_number)

        if level_number == 0:
            L = step.levels[level_number]
            P = L.prob
            P.write_solution(L.uend, L.time + L.dt)
