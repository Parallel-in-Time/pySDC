from __future__ import division
from pySDC.core.Hooks import hooks
import numpy as np
import matplotlib.pyplot as plt


class monitor(hooks):

    def __init__(self):
        """
        Initialization of particles output
        """
        super(monitor, self).__init__()

        self.init_radius = None

        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        plt.ion()
        self.im = None

    def pre_run(self, step, level_number):
        """
        Overwrite standard pre run hook

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(monitor, self).pre_run(step, level_number)
        L = step.levels[0]

        c = np.count_nonzero(L.u[0].values > 0.5)
        radius = np.sqrt(c / np.pi) * L.prob.dx

        radius1 = 0
        rows, cols = np.where(L.u[0].values > 0.5)
        for r in rows:
            radius1 = max(radius1, 0.5 - L.prob.xvalues[r])

        print(radius, radius1)
        self.init_radius = radius1
        # exit()

        self.add_to_stats(process=step.status.slot, time=L.time, level=-1, iter=step.status.iter,
                          sweep=L.status.sweep, type='radius', value=radius)
        # self.im = self.ax.imshow(L.u[0].values, vmin=-1.0, vmax=1.0)
        # plt.pause(0.001)
        # plt.show()

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

        # c = np.count_nonzero(L.uend.values > 0)
        # radius = np.sqrt(c / np.pi) * L.prob.dx
        radius = 0
        rows, cols = np.where(L.u[0].values > 0.5)
        for r in rows:
            radius = max(radius, 0.5 - L.prob.xvalues[r])

        print(radius, np.sqrt(self.init_radius ** 2 - 2.0 * L.time))

        self.add_to_stats(process=step.status.slot, time=L.time, level=-1, iter=step.status.iter,
                          sweep=L.status.sweep, type='radius', value=radius)

        # self.im.set_data(L.uend.values)
        # self.sframe = self.ax.imshow(L.uend.values, vmin=-1.0, vmax=1.0)
        # plt.pause(0.001)
        # plt.show()

        return None
