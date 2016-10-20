from __future__ import division
from pySDC.Hooks import hooks
from pySDC.Stats import stats

import matplotlib.pyplot as plt
import numpy as np

class particles_output(hooks):

    def __init__(self):
        """
        Initialization of particles output
        """
        super(particles_output,self).__init__()

        # add figure object for further use
        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        self.ax.set_xlim([-1.5,1.5])
        self.ax.set_ylim([-1.5,1.5])
        plt.ion()
        self.sframe = None



    def post_step(self, status):
        """
        Overwrite standard dump per step

        Args:
            status: status object per step
        """
        super(particles_output,self).post_step(status)

        # some abbreviations
        L = self.level
        u = L.uend

        R = np.linalg.norm(u.pos.values)
        H = 1/2*np.dot(u.vel.values,u.vel.values)+0.02/R

        stats.add_to_stats(step=status.step, time=status.time, type='energy', value=H)

        oldcol = self.sframe
        # self.sframe = self.ax.scatter(L.uend.pos.values[0],L.uend.pos.values[1],L.uend.pos.values[2])
        self.sframe = self.ax.scatter(L.uend.pos.values[0],L.uend.pos.values[1])
        # Remove old line collection before drawing
        if oldcol is not None:
            self.ax.collections.remove(oldcol)
        plt.pause(0.00001)

        return None
