from __future__ import division
from pySDC.Hooks import hooks
from pySDC.Stats import stats

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class plot_solution(hooks):

    def __init__(self):
        """
        Initialization of output
        """
        super(plot_solution,self).__init__()

        # add figure object for further use
        self.fig = plt.figure(figsize=(8,8))

    def dump_step(self,status):
        """
        Overwrite standard dump per step

        Args:
            status: status object per step
        """
        super(plot_solution,self).dump_step(status)

        yplot = self.level.uend.values
        xx    = self.level.prob.xx
        zz    = self.level.prob.zz

        self.fig.clear()
        ax = self.fig.gca(projection='3d')
        ax.view_init(elev=0., azim=-90.)
        surf = ax.plot_surface(xx, zz, yplot[2,:,:], rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_xlim(left   =  self.level.prob.x_b[0], right = self.level.prob.x_b[1])
        ax.set_ylim(bottom =  self.level.prob.z_b[0], top   = self.level.prob.z_b[1])
        ax.set_zlim(bottom = -1.0, top   = 1.0)
        plt.xlabel('x')
        plt.ylabel('z')
        plt.draw()
        plt.pause(0.00001)

        return None
