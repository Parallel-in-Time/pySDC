import numpy as np

from pySDC.core.hooks import Hooks


class monitor(Hooks):
    phase_thresh = 0.0  # count everything above this threshold to the high phase.

    def __init__(self):
        """
        Initialization of Allen-Cahn monitoring
        """
        super().__init__()

        self.init_radius = None

    def get_exact_radius(self, t):
        return np.sqrt(max(self.init_radius**2 - 2.0 * t, 0))

    @classmethod
    def get_radius(cls, u, dx):
        c = np.count_nonzero(u > cls.phase_thresh)
        return np.sqrt(c / np.pi) * dx

    @staticmethod
    def get_interface_width(u, L):
        # TODO: How does this generalize to different phase transitions?
        rows1 = np.where(u[L.prob.init[0][0] // 2, : L.prob.init[0][0] // 2] > -0.99)
        rows2 = np.where(u[L.prob.init[0][0] // 2, : L.prob.init[0][0] // 2] < 0.99)

        return (rows2[0][-1] - rows1[0][0]) * L.prob.dx / L.prob.eps

    def pre_run(self, step, level_number):
        """
        Record radius of the blob, exact radius and interface width.

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().pre_run(step, level_number)
        L = step.levels[0]

        radius = self.get_radius(L.u[0], L.prob.dx)
        interface_width = self.get_interface_width(L.u[0], L)
        self.init_radius = L.prob.radius

        if L.time == 0.0:
            self.add_to_stats(
                process=step.status.slot,
                time=L.time,
                level=-1,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type='computed_radius',
                value=radius,
            )
            self.add_to_stats(
                process=step.status.slot,
                time=L.time,
                level=-1,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type='exact_radius',
                value=self.init_radius,
            )
            self.add_to_stats(
                process=step.status.slot,
                time=L.time,
                level=-1,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type='interface_width',
                value=interface_width,
            )

    def post_step(self, step, level_number):
        """
        Record radius of the blob, exact radius and interface width.

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().post_step(step, level_number)

        # some abbreviations
        L = step.levels[0]

        radius = self.get_radius(L.uend, L.prob.dx)
        interface_width = self.get_interface_width(L.uend, L)

        exact_radius = self.get_exact_radius(L.time + L.dt)

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=-1,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='computed_radius',
            value=radius,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=-1,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='exact_radius',
            value=exact_radius,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=-1,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='interface_width',
            value=interface_width,
        )
