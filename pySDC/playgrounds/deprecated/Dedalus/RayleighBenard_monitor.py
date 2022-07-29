import numpy as np
import matplotlib.pyplot as plt


from pySDC.core.Hooks import hooks


class monitor(hooks):
    def __init__(self):
        super(monitor, self).__init__()
        self.imshow = None

    def pre_run(self, step, level_number):
        """
        Overwrite standard post step hook

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(monitor, self).pre_run(step, level_number)

        # some abbreviations
        L = step.levels[0]

        # self.imshow = plt.imshow(L.u[0].values[0]['g'])
        # # self.plt.colorbar()
        # # self.plt.pause(0.001)
        # plt.draw()
        # plt.pause(0.001)

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
        #
        # self.imshow.set_data(L.uend.values[0]['g'])
        # plt.draw()
        # plt.pause(0.0001)

        u_max = np.amax(abs(L.uend.values[1]['g']))

        print(u_max)
