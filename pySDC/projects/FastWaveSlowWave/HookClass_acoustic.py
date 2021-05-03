import numpy as np

from pySDC.core.Hooks import hooks


class dump_energy(hooks):
    def __init__(self):
        """
        Initialization of output
        """
        super(dump_energy, self).__init__()

        self.file = open('energy-sdc.txt', 'w')

    def post_step(self, step, level_number):
        """
        Default routine called after each iteration, compute and dump energy

        Args:
            step: the current step
            level_number: the current level number
        """

        super(dump_energy, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        xx = L.uend
        E = np.sum(np.square(xx[0, :]) + np.square(xx[1, :]))
        self.file.write('%30.20f\n' % E)
