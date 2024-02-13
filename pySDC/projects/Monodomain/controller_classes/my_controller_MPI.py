import numpy as np
from pySDC.core.Errors import ControllerError
from pySDC.implementations.controller_classes.controller_MPI import controller_MPI


class my_controller_MPI(controller_MPI):
    def __init__(self, controller_params, description, comm):
        super().__init__(controller_params, description, comm)

    def run(self, u0, t0, Tend):
        """
        Main driver for running the parallel version of SDC, MSSDC, MLSDC and PFASST

        Args:
            u0: initial values
            t0: starting time
            Tend: ending time

        Returns:
            end values on the finest level
            stats object containing statistics for each step, each level and each iteration
        """

        # reset stats to prevent double entries from old runs
        for hook in self.hooks:
            hook.reset_stats()

        # find active processes and put into new communicator
        rank = self.comm.Get_rank()
        num_procs = self.comm.Get_size()
        all_dt = self.comm.allgather(self.S.dt)
        all_time = [t0 + sum(all_dt[0:i]) for i in range(num_procs)]
        time = all_time[rank]
        all_active = [all_time[i] + all_dt[i] / 2.0 < Tend for i in range(num_procs)]

        if not any(all_active):
            raise ControllerError('Nothing to do, check t0, dt and Tend')

        active = all_active[rank]
        if not all(all_active):
            comm_active = self.comm.Split(active)
            rank = comm_active.Get_rank()
            num_procs = comm_active.Get_size()
        else:
            comm_active = self.comm

        self.S.status.slot = rank

        # initialize block of steps with u0
        self.restart_block(num_procs, time, u0, comm=comm_active)
        uend = u0

        # call post-setup hook
        for hook in self.hooks:
            hook.post_setup(step=None, level_number=None)

        # call pre-run hook
        for hook in self.hooks:
            hook.pre_run(step=self.S, level_number=0)

        comm_active.Barrier()

        # while any process still active...
        while active:
            while not self.S.status.done:
                self.pfasst(comm_active, num_procs)

            # determine where to restart
            restarts = comm_active.allgather(self.S.status.restart)
            restart_at = np.where(restarts)[0][0] if True in restarts else comm_active.size - 1

            # communicate time and solution to be used as next initial conditions
            if True in restarts:
                uend = self.S.levels[0].u[0].bcast(root=restart_at, comm=comm_active)
                tend = comm_active.bcast(self.S.time, root=restart_at)
                self.logger.info(f'Starting next block with initial conditions from step {restart_at}')
            else:
                time += self.S.dt
                uend = self.S.levels[0].uend.bcast(root=num_procs - 1, comm=comm_active)
                tend = comm_active.bcast(self.S.time + self.S.dt, root=comm_active.size - 1)

            # do convergence controller stuff
            if not self.S.status.restart:
                for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
                    C.post_step_processing(self, self.S)

            for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
                C.prepare_next_block(self, self.S, self.S.status.time_size, time, Tend, comm=comm_active)

            all_dt = comm_active.allgather(self.S.dt)
            all_time = [tend + sum(all_dt[0:i]) for i in range(num_procs)]

            time = all_time[rank]
            all_active = [all_time[i] + all_dt[i] / 2.0 < Tend for i in range(num_procs)]
            active = all_active[rank]
            if not all(all_active):
                comm_active = comm_active.Split(active)
                rank = comm_active.Get_rank()
                num_procs = comm_active.Get_size()
                self.S.status.slot = rank

            # initialize block of steps with u0
            self.restart_block(num_procs, time, uend, comm=comm_active)

        # call post-run hook
        for hook in self.hooks:
            hook.post_run(step=self.S, level_number=0)

        comm_active.Free()

        return uend, self.return_stats()

    def recv(self, target, source, tag=None, comm=None):
        """
        Wrapper around recv which updates the lmbda and yinf status
        """
        req = target.u[0].irecv(source=source, tag=tag, comm=comm)
        self.wait_with_interrupt(request=req)
        if self.S.status.force_done:
            return None
        # re-evaluate f on left interval boundary
        target.f[0] = target.prob.eval_f(target.u[0], target.time)
        target.sweep.update_lmbda_yinf_status(outdated=True)
