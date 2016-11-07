from __future__ import division
from pySDC.Hooks import hooks
import progressbar
import numpy as np


class particles_output(hooks):

    def __init__(self):
        """
        Initialization of particles output
        """
        super(particles_output, self).__init__()

    def pre_run(self, step, level_number):
        super(particles_output, self).pre_run(step, level_number)
        L = step.levels[0]
        self.bar_run = progressbar.ProgressBar(max_value=progressbar.UnknownLength)

    # def pre_step(self, step, level_number):
    #     super(particles_output, self).pre_step(step, level_number)
    #     L = step.levels[0]
    #     self.bar_step = progressbar.ProgressBar(max_value=-np.log10(L.params.restol))
    #
    # def post_iteration(self, step, level_number):
    #     super(particles_output, self).post_iteration(step, level_number)
    #     L = step.levels[0]
    #     self.bar_step.update(-np.log10(max(L.status.residual,L.params.restol)))

    def post_step(self, step, level_number):
        """
        Overwrite standard dump per step

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
