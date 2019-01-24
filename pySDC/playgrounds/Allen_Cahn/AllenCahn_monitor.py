import numpy as np

from pySDC.core.Hooks import hooks


class monitor(hooks):

    def __init__(self):
        """
        Initialization of Allen-Cahn monitoring
        """
        super(monitor, self).__init__()

        self.init_radius = None

    def pre_run(self, step, level_number):
        """
        Overwrite standard pre run hook

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(monitor, self).pre_run(step, level_number)
        L = step.levels[0]

        c = np.count_nonzero(L.u[0].values > 0.0)
        radius = np.sqrt(c / np.pi) * L.prob.dx

        radius1 = 0
        rows, cols = np.where(L.u[0].values > 0.0)
        for r in rows:
            radius1 = max(radius1, abs(L.prob.xvalues[r]))

        rows1 = np.where(L.u[0].values[int((L.prob.init[0]) / 2), :int((L.prob.init[0]) / 2)] > -0.99)
        rows2 = np.where(L.u[0].values[int((L.prob.init[0]) / 2), :int((L.prob.init[0]) / 2)] < 0.99)
        interface_width = (rows2[0][-1] - rows1[0][0]) * L.prob.dx / L.prob.params.eps

        self.init_radius = L.prob.params.radius

        self.add_to_stats(process=step.status.slot, time=L.time, level=-1, iter=step.status.iter,
                          sweep=L.status.sweep, type='computed_radius', value=radius)
        self.add_to_stats(process=step.status.slot, time=L.time, level=-1, iter=step.status.iter,
                          sweep=L.status.sweep, type='exact_radius', value=self.init_radius)
        self.add_to_stats(process=step.status.slot, time=L.time, level=-1, iter=step.status.iter,
                          sweep=L.status.sweep, type='interface_width', value=interface_width)

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

        c = np.count_nonzero(L.uend.values >= 0.0)
        radius = np.sqrt(c / np.pi) * L.prob.dx

        exact_radius = np.sqrt(max(self.init_radius ** 2 - 2.0 * (L.time + L.dt), 0))
        rows1 = np.where(L.uend.values[int((L.prob.init[0]) / 2), :int((L.prob.init[0]) / 2)] > -0.99)
        rows2 = np.where(L.uend.values[int((L.prob.init[0]) / 2), :int((L.prob.init[0]) / 2)] < 0.99)
        interface_width = (rows2[0][-1] - rows1[0][0]) * L.prob.dx / L.prob.params.eps

        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=-1, iter=step.status.iter,
                          sweep=L.status.sweep, type='computed_radius', value=radius)
        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=-1, iter=step.status.iter,
                          sweep=L.status.sweep, type='exact_radius', value=exact_radius)
        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=-1, iter=step.status.iter,
                          sweep=L.status.sweep, type='interface_width', value=interface_width)
