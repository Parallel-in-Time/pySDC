from pySDC.core.Hooks import hooks

class log_data_battery(hooks):

    def post_step(self, step, level_number):

        super(log_data_battery, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(process=step.status.slot, time=L.time+L.dt, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='current L', value=L.uend[0])
        self.add_to_stats(process=step.status.slot, time=L.time+L.dt, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='voltage C', value=L.uend[1])
        self.increment_stats(process=step.status.slot, time=L.time+L.dt, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='restart', value=1, initialize=0)
        self.add_to_stats(process=step.status.slot, time=L.time+L.dt, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='dt', value=L.dt)
