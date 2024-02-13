import itertools
import numpy as np
from pySDC.core.Errors import ControllerError, CommunicationError
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI


class my_controller_nonMPI(controller_nonMPI):
    def __init__(self, num_procs, controller_params, description):
        super().__init__(num_procs, controller_params, description)

    def run(self, u0, t0, Tend):
        """
        Main driver for running the serial version of SDC, MSSDC, MLSDC and PFASST (virtual parallelism)

        Args:
           u0: initial values
           t0: starting time
           Tend: ending time

        Returns:
            end values on the finest level
            stats object containing statistics for each step, each level and each iteration
        """

        # some initializations and reset of statistics
        uend = None
        num_procs = len(self.MS)
        for hook in self.hooks:
            hook.reset_stats()

        # initial ordering of the steps: 0,1,...,Np-1
        slots = list(range(num_procs))

        # initialize time variables of each step
        time = [t0 + sum(self.MS[j].dt for j in range(p)) for p in slots]

        # determine which steps are still active (time < Tend)
        # active = [time[p] < Tend - 10 * np.finfo(float).eps for p in slots] # with this there's the risk to overshoot Tend
        active = [time[p] + self.MS[p].dt / 2.0 < Tend - 10 * np.finfo(float).eps for p in slots]

        if not any(active):
            raise ControllerError('Nothing to do, check t0, dt and Tend.')

        # compress slots according to active steps, i.e. remove all steps which have times above Tend
        active_slots = list(itertools.compress(slots, active))

        # initialize block of steps with u0
        self.restart_block(active_slots, time, u0)

        for hook in self.hooks:
            hook.post_setup(step=None, level_number=None)

        # call pre-run hook
        for S in self.MS:
            for hook in self.hooks:
                hook.pre_run(step=S, level_number=0)

        # main loop: as long as at least one step is still active (time < Tend), do something
        while any(active):
            MS_active = [self.MS[p] for p in active_slots]
            done = False
            while not done:
                done = self.pfasst(MS_active)

            restarts = [S.status.restart for S in MS_active]
            restart_at = np.where(restarts)[0][0] if True in restarts else len(MS_active)
            if True in restarts:  # restart part of the block
                # initial condition to next block is initial condition of step that needs restarting
                uend = self.MS[restart_at].levels[0].u[0]
                time[active_slots[0]] = time[restart_at]
                self.logger.info(f'Starting next block with initial conditions from step {restart_at}')

            else:  # move on to next block
                # initial condition for next block is last solution of current block
                uend = self.MS[active_slots[-1]].levels[0].uend
                time[active_slots[0]] = time[active_slots[-1]] + self.MS[active_slots[-1]].dt

            for S in MS_active[:restart_at]:
                for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
                    C.post_step_processing(self, S)

            for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
                [C.prepare_next_block(self, S, len(active_slots), time, Tend, MS=MS_active) for S in self.MS]

            # setup the times of the steps for the next block
            for i in range(1, len(active_slots)):
                time[active_slots[i]] = time[active_slots[i] - 1] + self.MS[active_slots[i] - 1].dt

            # determine new set of active steps and compress slots accordingly
            # active = [time[p] < Tend - 10 * np.finfo(float).eps for p in slots]  # with this there's the risk to overshoot Tend
            active = [time[p] + self.MS[p].dt / 2.0 < Tend - 10 * np.finfo(float).eps for p in slots]

            active_slots = list(itertools.compress(slots, active))

            # restart active steps (reset all values and pass uend to u0)
            self.restart_block(active_slots, time, uend)

        # call post-run hook
        for S in self.MS:
            for hook in self.hooks:
                hook.post_run(step=S, level_number=0)

        return uend, self.return_stats()

    def recv_full(self, S, level=None, add_to_stats=False):
        """
        Wrapper around recv which updates the lmbda and yinf statu
        """

        def recv(target, source, tag=None):
            if tag is not None and source.tag != tag:
                raise CommunicationError('source and target tag are not the same, got %s and %s' % (source.tag, tag))
            # simply do a deepcopy of the values uend to become the new u0 at the target
            target.u[0] = target.prob.dtype_u(source.uend)
            # re-evaluate f on left interval boundary
            target.f[0] = target.prob.eval_f(target.u[0], target.time)
            target.sweep.update_lmbda_yinf_status(outdated=True)

        for hook in self.hooks:
            hook.pre_comm(step=S, level_number=level)
        if not S.status.prev_done and not S.status.first:
            self.logger.debug('Process %2i receives from %2i on level %2i with tag %s' % (S.status.slot, S.prev.status.slot, level, S.status.iter))
            recv(S.levels[level], S.prev.levels[level], tag=(level, S.status.iter, S.prev.status.slot))
        for hook in self.hooks:
            hook.post_comm(step=S, level_number=level, add_to_stats=add_to_stats)
