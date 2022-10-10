from matplotlib import cm
from matplotlib import pyplot as plt

from pySDC.core.Hooks import hooks


class plot_solution(hooks):
    def __init__(self):
        """
        Initialization of output
        """
        super(plot_solution, self).__init__()

        # add figure object for further use
        #        self.fig = plt.figure(figsize=(18,6))
        self.fig = plt.figure(figsize=(9, 9))

        self.counter = 0

    def post_step(self, status):
        """
        Overwrite standard dump per step

        Args:
            status: status object per step
        """
        super(plot_solution, self).post_step(status)

        if False:
            yplot = self.level.uend.values
            xx = self.level.prob.xc
            yy = self.level.prob.yc
            self.fig.clear()
            plt.plot(xx[:, 0], yplot[0, :, 0])
            plt.ylim([-1.0, 1.0])
            plt.show(block=False)
            plt.pause(0.00001)

        if True:
            yplot = self.level.uend.values
            xx = self.level.prob.xc
            zz = self.level.prob.yc
            self.fig.clear()
            CS = plt.contourf(
                xx, zz, yplot[0, :, :], rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False
            )
            cbar = plt.colorbar(CS)
            # plt.axes().set_xlim(xmin = self.level.prob.x_b[0], xmax = self.level.prob.x_b[1])
            # plt.axes().set_ylim(ymin = self.level.prob.z_b[0], ymax = self.level.prob.z_b[1])
            # plt.axes().set_aspect('equal')
            plt.xlabel('x')
            plt.ylabel('z')
            # plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.00001)

        return None
