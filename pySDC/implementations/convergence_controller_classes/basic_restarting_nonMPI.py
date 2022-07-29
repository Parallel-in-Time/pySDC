import numpy as np
from pySDC.core.ConvergenceController import ConvergenceController


class BasicRestartingNonMPI(ConvergenceController):
    def setup(self, controller, params, description):
        return {'control_order': 100, **params}

    def prepare_next_block_nonMPI(self, controller, MS, active_slots, time, Tend):
        """
        Spread the step size of the last step with no restarted predecessors to all steps
        """
        # figure out where the block is restarted
        restarts = [MS[p].status.restart for p in active_slots]
        if True in restarts:
            restart_at = np.where(restarts)[0][0]
        else:
            restart_at = len(restarts) - 1

        # Compute the maximum allowed step size based on Tend.
        dt_max = (Tend - time[0]) / len(active_slots)

        # record the step sizes to restart with from all the levels of the step
        new_steps = [None] * len(MS[restart_at].levels)
        for i in range(len(MS[restart_at].levels)):
            l = MS[restart_at].levels[i]
            # overrule the step size control to reach Tend if needed
            new_steps[i] = min(
                [l.status.dt_new if l.status.dt_new is not None else l.params.dt, max([dt_max, l.params.dt_initial])]
            )

        # spread the step sizes to all levels
        for j in range(len(active_slots)):
            # get slot number
            p = active_slots[j]

            for i in range(len(MS[p].levels)):
                MS[p].levels[i].params.dt = new_steps[i]

    def determine_restart(self, controller, S):
        self.restart = S.status.restart or self.restart
        S.status.restart = S.status.restart or self.restart

    def reset_global_variables_nonMPI(self, controller):
        self.restart = False
