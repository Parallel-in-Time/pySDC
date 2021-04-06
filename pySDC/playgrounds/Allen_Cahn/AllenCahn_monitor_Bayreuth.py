import numpy as np

from pySDC.core.Hooks import hooks


class monitor(hooks):

    def __init__(self):
        """
        Initialization of Allen-Cahn monitoring
        """
        super(monitor, self).__init__()

        self.init_volume = None

    def pre_run(self, step, level_number):
        """
        Overwrite standard pre run hook

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(monitor, self).pre_run(step, level_number)
        L = step.levels[0]

        self.init_volume = L.prob.dx * sum(L.u[0])

        print(self.init_volume)

        self.add_to_stats(process=step.status.slot, time=L.time, level=-1, iter=step.status.iter,
                          sweep=L.status.sweep, type='exact_volume', value=self.init_volume)

    def post_step(self, step, level_number):
        """
        Overwrite standard post step hook

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(monitor, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[0]

        computed_volume = L.prob.dx * sum(L.uend)

        uex = L.prob.u_exact(L.time + L.dt)

        exact_volume = L.prob.dx * sum(uex)

        print(exact_volume, computed_volume, abs(exact_volume - computed_volume), abs(L.uend - uex))

        self.add_to_stats(process=step.status.slot, time=L.time, level=-1, iter=step.status.iter,
                          sweep=L.status.sweep, type='exact_volume', value=exact_volume)
        self.add_to_stats(process=step.status.slot, time=L.time, level=-1, iter=step.status.iter,
                          sweep=L.status.sweep, type='computed_volume', value=computed_volume)
