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
        #        self.fig = plt.figure(figsize=(18,6))
        #self.fig = plt.figure(figsize=(15,5))
        #self.counter = 0

    def post_sweep(self, status):
        """
        Set new GMRES tolerance depending on the previous SDC residual

        Args:
            status: status object per step
        """
        super(plot_solution,self).post_sweep(status)
        self.level.prob.gmres_tol = max(self.level.status.residual*self.level.prob.gmres_tol_factor,self.level.prob.gmres_tol_limit)

    def pre_iteration(self, status):
        """
        Set new GMRES tolerance depending on the initial SDC residual

        Args:
            status: status object per step
        """
        super(plot_solution,self).pre_iteration(status)
        self.level.sweep.compute_residual()
        self.level.prob.gmres_tol = max(self.level.status.residual*self.level.prob.gmres_tol_factor,self.level.prob.gmres_tol_limit)

    def post_step(self, status):
        """
        Overwrite standard dump per step

        Args:
            status: status object per step
        """
        super(plot_solution,self).post_step(status)

        #yplot = self.level.uend.values
        #xx    = self.level.prob.xx
        #zz    = self.level.prob.zz
        #self.fig.clear()
        #plt.plot( xx[:,0], yplot[2,:,0])
        #plt.ylim([-1.1, 1.1])
        #plt.show(block=False)
        #plt.pause(0.00001)        
        
        fs = 18
        
        if False:
          yplot = self.level.uend.values
          xx    = self.level.prob.xx
          zz    = self.level.prob.zz
          #self.fig.clear()
          levels = [-5e-3, -4e-3, -3e-3, -2e-3, -1e3, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3]
          CS = plt.contourf(xx, zz, yplot[2,:,:], rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#          CS = plt.contour(xx, zz, yplot[2,:,:], rstride=1, cstride=1, linewidth=0, antialiased=False)
#          cbar = plt.colorbar(CS)
          plt.axes().set_xlim(xmin = self.level.prob.x_bounds[0], xmax = self.level.prob.x_bounds[1])
          plt.axes().set_ylim(ymin = self.level.prob.z_bounds[0], ymax = self.level.prob.z_bounds[1])
          plt.axes().set_aspect('auto')
          plt.xlabel('x in km', fontsize=fs)
          plt.ylabel('z in km', fontsize=fs)
          plt.tick_params(axis='both', which='major', labelsize=fs)
          #plt.tight_layout()
          #plt.show(block=False)
          #plt.show()
          plt.draw()
          plt.gcf().savefig('images/boussinesq-'+"%04d" % self.counter +'.png', bbox_inches='tight')
          self.counter = self.counter + 1
          plt.pause(0.001)
        
          # NOTE: Can use ffmpeg to collate images into movie.
          # HELPFUL WEBSITE: http://hamelot.co.uk/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/
          # USEFUL COMMAND: ffmpeg -r 25 -i boussinesq-%04d.jpeg -vcodec libx264 -crf 25 test1800.avi
          # WEBSITE TO CONVERT TO MP4: http://video.online-convert.com/convert-to-mp4
        return None
