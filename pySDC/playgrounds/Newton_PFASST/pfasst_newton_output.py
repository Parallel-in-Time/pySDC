from __future__ import division
from pySDC.core.Hooks import hooks
import numpy as np
import pySDC.helpers.plot_helper as plt_helper


class output(hooks):

    def post_iteration(self, step, level_number):
        """
        Overwrite standard post step hook

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(output, self).post_iteration(step, level_number)

        # some abbreviations
        L = step.levels[0]
        P = L.prob

        print(step.status.iter, P.inner_solve_counter)

