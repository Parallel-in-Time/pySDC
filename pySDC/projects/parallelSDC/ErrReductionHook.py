from pySDC.core.Hooks import hooks
import numpy as np


class err_reduction_hook(hooks):

    def pre_iteration(self, step, level_number):
        """
        Routine called before iteration starts

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(err_reduction_hook, self).pre_iteration(step, level_number)

        L = step.levels[level_number]
        if step.status.iter == 2 and np.isclose(L.time + L.dt, 0.1):

            P = L.prob

            err = []
            for m in range(L.sweep.coll.num_nodes):
                uex = P.u_exact(L.time + L.dt * L.sweep.coll.nodes[m])
                err.append(abs(uex - L.u[m + 1]))
            err_full = max(err)
            self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=step.status.iter,
                              sweep=L.status.sweep, type='error_pre_iteration', value=err_full)
            # print(L.time, step.status.iter, err_full)

    def post_iteration(self, step, level_number):
        """
        Routine called after each iteration

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(err_reduction_hook, self).post_iteration(step, level_number)

        L = step.levels[level_number]

        if step.status.iter == 2 and np.isclose(L.time + L.dt, 0.1):

            P = L.prob

            err = []
            for m in range(L.sweep.coll.num_nodes):
                uex = P.u_exact(L.time + L.dt * L.sweep.coll.nodes[m])
                err.append(abs(uex - L.u[m + 1]))
            err_full = max(err)
            self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=step.status.iter,
                              sweep=L.status.sweep, type='error_post_iteration', value=err_full)
            # print(L.time, step.status.iter, err_full)
