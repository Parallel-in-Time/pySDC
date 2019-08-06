from pySDC.core.Hooks import hooks
import numpy as np


class error_est(hooks):
    """
    Hook class to add output of error
    """

    def __init__(self):

        super(error_est, self).__init__()

        self.diff_old = None
        self.diff_first = None
        self.eta = 1E-04

    def pre_step(self, step, level_number):
        super(error_est, self).pre_step(step, level_number)
        # some abbreviations
        L = step.levels[level_number]
        self.diff_old = 1.0


    def pre_iteration(self, step, level_number):
        super(error_est, self).pre_iteration(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.uold[:] = L.u[:]

    def post_sweep(self, step, level_number):
        """
        Default routine called after each iteration
        Args:
            step: the current step
            level_number: the current level number
        """

        super(error_est, self).post_sweep(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        diff_new = 0.0
        for m in range(1, L.sweep.coll.num_nodes + 1):
            diff_new = max(diff_new, abs(L.uold[m] - L.u[m]))

        Ltilde = diff_new / self.diff_old

        self.diff_old = diff_new

        if step.status.iter == 1:
            self.diff_first = diff_new
        elif step.status.iter > 1:
            Kest = np.log(self.eta * (1 - Ltilde) / self.diff_first) / np.log(Ltilde)
            if np.ceil(Kest) <= step.status.iter:
                step.status.force_done = True
            # errest = (self.eta - Ltilde ** Kest * self.diff_first / (1 - Ltilde)) / (Ltilde ** (Kest - step.status.iter - 1))
            # print(step.status.iter, errest, np.ceil(Kest))
            print(step.status.iter, np.ceil(Kest))
            # L.u[2].values[:] += self.eta / 2


    def post_step(self, step, level_number):
        """
        Default routine called after each step
        Args:
            step: the current step
            level_number: the current level number
        """

        super(error_est, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]
        P = L.prob

        L.sweep.compute_end_point()

        uex = P.u_exact(step.time + step.dt)
        err = abs(uex - L.uend)

        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=step.status.iter,
                          sweep=L.status.sweep, type='error_after_step', value=err)
