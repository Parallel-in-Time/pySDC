import numpy as np

# import progressbar

from pySDC.core.Hooks import hooks


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

        # if hasattr(L.prob, 'Tend'):
        #     self.bar_run = progressbar.ProgressBar(max_value=L.prob.Tend)
        # else:
        #     self.bar_run = progressbar.ProgressBar(max_value=progressbar.UnknownLength)

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

        # self.bar_run.update(L.time)

        R = np.linalg.norm(u.pos)
        H = 1 / 2 * np.dot(u.vel[:, 0], u.vel[:, 0]) + L.prob.a0 / R

        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=-1,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='energy',
            value=H,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=-1,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='position',
            value=L.uend.pos,
        )

        return None
