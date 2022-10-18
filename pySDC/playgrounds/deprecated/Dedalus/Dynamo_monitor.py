import numpy as np

from pySDC.core.Hooks import hooks


class monitor(hooks):
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

        bx_max = np.amax(abs(L.u[0][..., 0]))
        # bx_max = np.amax(abs(L.u[0].values[0]['g']))

        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=-1,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='bx_max',
            value=bx_max,
        )

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

        bx_max = np.amax(abs(L.uend[..., 0]))
        # bx_max = np.amax(abs(L.uend.values[0]['g']))

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=-1,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='bx_max',
            value=bx_max,
        )
