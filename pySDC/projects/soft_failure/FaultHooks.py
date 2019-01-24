from __future__ import division

import numpy as np

from pySDC.core.Hooks import hooks


class fault_hook(hooks):

    def __init__(self):
        """
        Initialization of fault hooks
        """
        super(fault_hook, self).__init__()

        self.fault_iteration = None

    def pre_run(self, step, level_number):

        super(fault_hook, self).pre_run(step, level_number)

        L = step.levels[level_number]

        L.sweep.reset_fault_stats()

        self.fault_iteration = np.random.randint(1, L.sweep.params.niters)

    def pre_iteration(self, step, level_number):

        super(fault_hook, self).pre_iteration(step, level_number)

        L = step.levels[level_number]

        L.sweep.fault_iteration = self.fault_iteration == step.status.iter

    def post_run(self, step, level_number):

        super(fault_hook, self).post_run(step, level_number)

        L = step.levels[level_number]

        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=step.status.iter,
                          sweep=L.status.sweep, type='fault_stats', value=L.sweep.fault_stats)
