
from pySDC.core.Hooks import hooks


class particles_output(hooks):
    def __init__(self):
        """
        Initialization of particles output
        """
        super(particles_output, self).__init__()

    def pre_run(self, step, level_number):
        """
        Overwrite default routine called before time-loop starts
        Args:
            step: the current step
            level_number: the current level number
        """
        super(particles_output, self).pre_run(step, level_number)

    def post_step(self, step, level_number):
        """
        Default routine called after each iteration
        Args:
            step: the current step
            level_number: the current level number
        """

        super(particles_output, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        # self.bar_run.update(L.time)

        L.sweep.compute_end_point()
        part = L.uend
        part_exact=L.prob.u_exact(L.time + L.dt)
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='position',
            value=part.pos,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='velocity',
            value=part.vel,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='position_exact',
            value=part_exact.pos,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='velocity_exact',
            value=part_exact.vel,
        )
