import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import progressbar

from pySDC.core.Hooks import hooks


class particles_output(hooks):

    def __init__(self):
        """
        Initialization of particles output
        """
        super(particles_output, self).__init__()

        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')
        self.ax.set_xlim3d([-20, 20])
        self.ax.set_ylim3d([-20, 20])
        self.ax.set_zlim3d([-20, 20])
        plt.ion()
        self.sframe = None

        self.bar_run = None

    def pre_run(self, step, level_number):
        """
        Overwrite default routine called before time-loop starts
        Args:
            step: the current step
            level_number: the current level number
        """
        super(particles_output, self).pre_run(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        if hasattr(L.prob.params, 'Tend'):
            self.bar_run = progressbar.ProgressBar(max_value=L.prob.params.Tend)
        else:
            self.bar_run = progressbar.ProgressBar(max_value=progressbar.UnknownLength)

        part = L.u[0]
        N = L.prob.params.nparts
        w = np.array([1, 1, -2])

        # compute (slowly..) the potential at u0
        fpot = np.zeros(N)
        for i in range(N):
            # inner loop, omit ith particle
            for j in range(0, i):
                dist2 = np.linalg.norm(part.pos.values[:, i] - part.pos.values[:, j], 2) ** 2 + L.prob.params.sig ** 2
                fpot[i] += part.q[j] / np.sqrt(dist2)
            for j in range(i + 1, N):
                dist2 = np.linalg.norm(part.pos.values[:, i] - part.pos.values[:, j], 2) ** 2 + L.prob.params.sig ** 2
                fpot[i] += part.q[j] / np.sqrt(dist2)
            fpot[i] -= L.prob.params.omega_E ** 2 * part.m[i] / part.q[i] / 2.0 * \
                np.dot(w, part.pos.values[:, i] * part.pos.values[:, i])

        # add up kinetic and potntial contributions to total energy
        epot = 0
        ekin = 0
        for n in range(N):
            epot += part.q[n] * fpot[n]
            ekin += part.m[n] / 2.0 * np.dot(part.vel.values[:, n], part.vel.values[:, n])

        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='etot', value=epot + ekin)

    def post_step(self, step, level_number):
        """
        Default routine called after each iteration
        Args:
            step: the current step
            level_number: the current level number
        """

        super(particles_output, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        self.bar_run.update(L.time)

        L.sweep.compute_end_point()
        part = L.uend
        N = L.prob.params.nparts
        w = np.array([1, 1, -2])

        # compute (slowly..) the potential at uend
        fpot = np.zeros(N)
        for i in range(N):
            # inner loop, omit ith particle
            for j in range(0, i):
                dist2 = np.linalg.norm(part.pos.values[:, i] - part.pos.values[:, j], 2) ** 2 + L.prob.params.sig ** 2
                fpot[i] += part.q[j] / np.sqrt(dist2)
            for j in range(i + 1, N):
                dist2 = np.linalg.norm(part.pos.values[:, i] - part.pos.values[:, j], 2) ** 2 + L.prob.params.sig ** 2
                fpot[i] += part.q[j] / np.sqrt(dist2)
            fpot[i] -= L.prob.params.omega_E ** 2 * part.m[i] / part.q[i] / 2.0 * \
                np.dot(w, part.pos.values[:, i] * part.pos.values[:, i])

        # add up kinetic and potntial contributions to total energy
        epot = 0
        ekin = 0
        for n in range(N):
            epot += part.q[n] * fpot[n]
            ekin += part.m[n] / 2.0 * np.dot(part.vel.values[:, n], part.vel.values[:, n])

        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=step.status.iter,
                          sweep=L.status.sweep, type='etot', value=epot + ekin)

        oldcol = self.sframe
        # # self.sframe = self.ax.scatter(L.uend.pos.values[0],L.uend.pos.values[1],L.uend.pos.values[2])
        self.sframe = self.ax.scatter(L.uend.pos.values[0::3], L.uend.pos.values[1::3], L.uend.pos.values[2::3])
        # Remove old line collection before drawing
        if oldcol is not None:
            self.ax.collections.remove(oldcol)
        plt.pause(0.001)

        return None
