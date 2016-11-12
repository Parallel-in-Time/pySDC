from __future__ import division
from pySDC.core.Hooks import hooks
import progressbar
import numpy as np


class particles_output(hooks):

    def __init__(self):
        """
        Initialization of particles output
        """
        super(particles_output, self).__init__()
        self.bar_run = None

    def pre_run(self, step, level_number):
        """
        Overwrite standard pre run hook

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(particles_output, self).pre_run(step, level_number)
        L = step.levels[0]

        if hasattr(L.prob.params, 'Tend'):
            self.bar_run = progressbar.ProgressBar(max_value=L.prob.params.Tend)
        else:
            self.bar_run = progressbar.ProgressBar(max_value=progressbar.UnknownLength)

    def post_step(self, step, level_number):
        """
        Overwrite standard post step hook

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(particles_output, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[0]
        u = L.uend

        self.bar_run.update(L.time)

        R = np.linalg.norm(u.pos.values)
        H = 1 / 2 * np.dot(u.vel.values, u.vel.values) + L.prob.params.a0 / R

        self.add_to_stats(process=step.status.slot, time=L.time, level=-1, iter=step.status.iter,
                          type='energy', value=H)

        self.add_to_stats(process=step.status.slot, time=L.time, level=-1, iter=step.status.iter,
                          type='position', value=L.uend.pos.values)

        return None
