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
        self.ax_cut = fig.add_subplot(111)
        plt.ion()
        self.im = None
        self.im_cut = None
        self.im_rad = None

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

        rows1 = np.where(L.u[0].values[int((L.prob.init[0])/2), :int((L.prob.init[0])/2)] > -1 + L.prob.params.eps)
        rows2 = np.where(L.u[0].values[int((L.prob.init[0])/2), :int((L.prob.init[0])/2)] < 1 - L.prob.params.eps)
        # print(rows1[0], rows2[0])
        print((rows2[0][-1] - rows1[0][0]) * L.prob.dx / L.prob.params.eps)
        # exit()
        print(radius, radius1)
        self.init_radius = L.prob.params.radius
        # exit()

        self.add_to_stats(process=step.status.slot, time=L.time, level=-1, iter=step.status.iter,
                          sweep=L.status.sweep, type='radius', value=radius)
        # self.im = self.ax.imshow(L.u[0].values, vmin=-1.0, vmax=1.0)
        # self.im_cut = self.ax_cut.plot(L.prob.xvalues, L.u[0].values[int((L.prob.init[0]-1)/2), :])
        self.im_rad = self.ax.text(0,0,'')
        self.ax.set_xlim(0, 0.05)
        self.ax.set_ylim(0, 0.3)
        plt.pause(0.001)
        plt.show()

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
        radius1 = 0
        rows = np.where(L.uend.values[int((L.prob.init[0]-1)/2), :] >= 0.0)
        for r in rows[0]:
            radius1 = max(radius, abs(L.prob.xvalues[r]))
        radius_exact = np.sqrt(max(self.init_radius ** 2 - 2.0 * (L.time + L.dt), 0))
        print(radius, radius1, radius_exact)

        self.add_to_stats(process=step.status.slot, time=L.time, level=-1, iter=step.status.iter,
                          sweep=L.status.sweep, type='radius', value=radius)

        # self.im.set_data(L.uend.values)
        # self.im_cut = self.ax_cut.plot(L.prob.xvalues, L.uend.values[int((L.prob.init[0]-1)/2), :])
        # self.im_cut = self.ax_cut.text(0.5-radius_exact, 0.5, 'X')
        self.im_rad = self.ax.text(L.time + L.dt, radius, 'o', color='r')
        self.im_rad = self.ax.text(L.time + L.dt, radius1, 'o', color='b')
        self.im_rad = self.ax.text(L.time + L.dt, radius_exact, 'o', color='k')
        # self.sframe = self.ax.imshow(L.uend.values, vmin=-1.0, vmax=1.0)
        plt.pause(0.001)
        plt.show()

        return None

    def post_run(self, step, level_number):
        super(monitor, self).post_run(step, level_number)
        plt.show()
        plt.savefig('allen-cahn.png')
