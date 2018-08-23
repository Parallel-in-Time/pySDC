from __future__ import division
from pySDC.core.Hooks import hooks
import numpy as np
import pySDC.helpers.plot_helper as plt_helper


class libpfasst_output(hooks):

    def __init__(self):
        """
        Initialization of Allen-Cahn monitoring
        """
        super(libpfasst_output, self).__init__()

        self.step_counter = 1

    def pre_run(self, step, level_number):
        """
        Overwrite standard post step hook

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(libpfasst_output, self).pre_run(step, level_number)

        print()
        print('--- BEGIN RUN OUTPUT')

    def post_sweep(self, step, level_number):
        """
        Overwrite standard post step hook

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(libpfasst_output, self).post_iteration(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()
        uex = L.prob.u_exact(L.time + L.dt)
        err = abs(uex - L.uend)

        out = 'error: step: ' + str(self.step_counter).zfill(3)
        out += ' iter:  ' + str(step.status.iter).zfill(3) + ' level: ' + str(level_number + 1).zfill(2)
        out += ' error: % 10.7e' % err
        out += ' res: %12.10e' % L.status.residual

        print(out)

    def post_step(self, step, level_number):
        """
        Overwrite standard post step hook

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(libpfasst_output, self).post_step(step, level_number)

        self.step_counter += 1

    def post_run(self, step, level_number):
        """
        Overwrite standard post step hook

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(libpfasst_output, self).post_run(step, level_number)

        print('--- END RUN OUTPUT')
        print()


