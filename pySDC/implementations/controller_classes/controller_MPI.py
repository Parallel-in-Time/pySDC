import numpy as np
from mpi4py import MPI

from pySDC.core.Controller import controller
from pySDC.core.Errors import ControllerError
from pySDC.core.Step import step
from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestarting


class controller_MPI(controller):
    """

    PFASST controller, running parallel version of PFASST in blocks (MG-style)

    """

    def __init__(self, controller_params, description, comm):
        """
        Initialization routine for PFASST controller

        Args:
            controller_params: parameter set for the controller and the step class
            description: all the parameters to set up the rest (levels, problems, transfer, ...)
            comm: MPI communicator
        """

        # call parent's initialization routine
        super().__init__(controller_params, description, useMPI=True)

        # create single step per processor
        self.S = step(description)

        # pass communicator for future use
        self.comm = comm

        num_procs = self.comm.Get_size()
        rank = self.comm.Get_rank()

        # insert data on time communicator to the steps (helpful here and there)
        self.S.status.time_size = num_procs

        self.base_convergence_controllers += [BasicRestarting.get_implementation("MPI")]
        for convergence_controller in self.base_convergence_controllers:
            self.add_convergence_controller(convergence_controller, description)

        if self.params.dump_setup and rank == 0:
            self.dump_setup(step=self.S, controller_params=controller_params, description=description)

        num_levels = len(self.S.levels)

        # add request handler for status send
        self.req_status = None
        # add request handle container for isend
        self.req_send = [None] * num_levels
        self.req_ibcast = None
        self.req_diff = None

        if num_procs > 1 and num_levels > 1:
            for L in self.S.levels:
                if not L.sweep.coll.right_is_node or L.sweep.params.do_coll_update:
                    raise ControllerError("For PFASST to work, we assume uend^k = u_M^k")

        if num_levels == 1 and self.params.predict_type is not None:
            self.logger.warning(
                'you have specified a predictor type but only a single level.. predictor will be ignored'
            )

        for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
            C.setup_status_variables(self, comm=comm)

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
        all_active = all_time < Tend - 10 * np.finfo(float).eps

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
            all_active = all_time < Tend - 10 * np.finfo(float).eps
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

    def restart_block(self, size, time, u0, comm):
        """
        Helper routine to reset/restart block of (active) steps

        Args:
            size: number of active time steps
            time: current time
            u0: initial value to distribute across the steps
            comm: the communicator

        Returns:
            block of (all) steps
        """

        # store link to previous step
        self.S.prev = (self.S.status.slot - 1) % size
        self.S.next = (self.S.status.slot + 1) % size

        # resets step
        self.S.reset_step()
        # determine whether I am the first and/or last in line
        self.S.status.first = self.S.prev == size - 1
        self.S.status.last = self.S.next == 0
        # intialize step with u0
        self.S.init_step(u0)
        # reset some values
        self.S.status.done = False
        self.S.status.iter = 0
        self.S.status.stage = 'SPREAD'
        for l in self.S.levels:
            l.tag = None
        self.req_status = None
        self.req_diff = None
        self.req_ibcast = None
        self.req_diff = None
        self.req_send = [None] * len(self.S.levels)
        self.S.status.prev_done = False
        self.S.status.force_done = False

        for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
            C.reset_status_variables(self, comm=comm)

        self.S.status.time_size = size

        for lvl in self.S.levels:
            lvl.status.time = time
            lvl.status.sweep = 1

    def recv(self, target, source, tag=None, comm=None):
        """
        Receive function

        Args:
            target: level which will receive the values
            source: level which initiated the send
            tag: identifier to check if this message is really for me
            comm: communicator
        """
        req = target.u[0].irecv(source=source, tag=tag, comm=comm)
        self.wait_with_interrupt(request=req)
        if self.S.status.force_done:
            return None
        # re-evaluate f on left interval boundary
        target.f[0] = target.prob.eval_f(target.u[0], target.time)

    def send_full(self, comm=None, blocking=False, level=None, add_to_stats=False):
        """
        Function to perform the send, including bookkeeping and logging

        Args:
            comm: the communicator
            blocking: flag to indicate that we need blocking communication
            level: the level number
            add_to_stats: a flag to end recording data in the hooks (defaults to False)
        """
        for hook in self.hooks:
            hook.pre_comm(step=self.S, level_number=level)

        if not blocking:
            self.wait_with_interrupt(request=self.req_send[level])
            if self.S.status.force_done:
                return None

        self.S.levels[level].sweep.compute_end_point()

        if not self.S.status.last:
            self.logger.debug(
                'isend data: process %s, stage %s, time %s, target %s, tag %s, iter %s'
                % (
                    self.S.status.slot,
                    self.S.status.stage,
                    self.S.time,
                    self.S.next,
                    level * 100 + self.S.status.iter,
                    self.S.status.iter,
                )
            )
            self.req_send[level] = self.S.levels[level].uend.isend(
                dest=self.S.next, tag=level * 100 + self.S.status.iter, comm=comm
            )
            if blocking:
                self.wait_with_interrupt(request=self.req_send[level])
                if self.S.status.force_done:
                    return None

        for hook in self.hooks:
            hook.post_comm(step=self.S, level_number=level, add_to_stats=add_to_stats)

    def recv_full(self, comm, level=None, add_to_stats=False):
        """
        Function to perform the recv, including bookkeeping and logging

        Args:
            comm: the communicator
            level: the level number
            add_to_stats: a flag to end recording data in the hooks (defaults to False)
        """

        for hook in self.hooks:
            hook.pre_comm(step=self.S, level_number=level)
        if not self.S.status.first and not self.S.status.prev_done:
            self.logger.debug(
                'recv data: process %s, stage %s, time %s, source %s, tag %s, iter %s'
                % (
                    self.S.status.slot,
                    self.S.status.stage,
                    self.S.time,
                    self.S.prev,
                    level * 100 + self.S.status.iter,
                    self.S.status.iter,
                )
            )
            self.recv(target=self.S.levels[level], source=self.S.prev, tag=level * 100 + self.S.status.iter, comm=comm)

        for hook in self.hooks:
            hook.post_comm(step=self.S, level_number=level, add_to_stats=add_to_stats)

    def wait_with_interrupt(self, request):
        """
        Wrapper for waiting for the completion of a non-blocking communication, can be interrupted

        Args:
            request: request to wait for
        """
        if request is not None and self.req_ibcast is not None:
            while not request.Test():
                if self.req_ibcast.Test():
                    self.logger.debug(f'{self.S.status.slot} has been cancelled during {self.S.status.stage}..')
                    self.S.status.stage = f'CANCELLED_{self.S.status.stage}'
                    self.S.status.force_done = True
                    return None
        if request is not None:
            request.Wait()

    def check_iteration_estimate(self, comm):
        """
        Routine to compute and check error/iteration estimation

        Args:
            comm: time-communicator
        """

        # Compute diff between old and new values
        diff_new = 0.0
        L = self.S.levels[0]

        for m in range(1, L.sweep.coll.num_nodes + 1):
            diff_new = max(diff_new, abs(L.uold[m] - L.u[m]))

        # Send forward diff
        for hook in self.hooks:
            hook.pre_comm(step=self.S, level_number=0)

        self.wait_with_interrupt(request=self.req_diff)
        if self.S.status.force_done:
            return None

        if not self.S.status.first:
            prev_diff = np.empty(1, dtype=float)
            req = comm.Irecv((prev_diff, MPI.DOUBLE), source=self.S.prev, tag=999)
            self.wait_with_interrupt(request=req)
            if self.S.status.force_done:
                return None
            self.logger.debug(
                'recv diff: status %s, process %s, time %s, source %s, tag %s, iter %s'
                % (prev_diff, self.S.status.slot, self.S.time, self.S.prev, 999, self.S.status.iter)
            )
            diff_new = max(prev_diff[0], diff_new)

        if not self.S.status.last:
            self.logger.debug(
                'isend diff: status %s, process %s, time %s, target %s, tag %s, iter %s'
                % (diff_new, self.S.status.slot, self.S.time, self.S.next, 999, self.S.status.iter)
            )
            tmp = np.array(diff_new, dtype=float)
            self.req_diff = comm.Issend((tmp, MPI.DOUBLE), dest=self.S.next, tag=999)

        for hook in self.hooks:
            hook.post_comm(step=self.S, level_number=0)

        # Store values from first iteration
        if self.S.status.iter == 1:
            self.S.status.diff_old_loc = diff_new
            self.S.status.diff_first_loc = diff_new
        # Compute iteration estimate
        elif self.S.status.iter > 1:
            Ltilde_loc = min(diff_new / self.S.status.diff_old_loc, 0.9)
            self.S.status.diff_old_loc = diff_new
            alpha = 1 / (1 - Ltilde_loc) * self.S.status.diff_first_loc
            Kest_loc = np.log(self.S.params.errtol / alpha) / np.log(Ltilde_loc) * 1.05  # Safety factor!
            self.logger.debug(
                f'LOCAL: {L.time:8.4f}, {self.S.status.iter}: {int(np.ceil(Kest_loc))}, '
                f'{Ltilde_loc:8.6e}, {Kest_loc:8.6e}, '
                f'{Ltilde_loc ** self.S.status.iter * alpha:8.6e}'
            )
            Kest_glob = Kest_loc
            # If condition is met, send interrupt
            if np.ceil(Kest_glob) <= self.S.status.iter:
                if self.S.status.last:
                    self.logger.debug(f'{self.S.status.slot} is done, broadcasting..')
                    for hook in self.hooks:
                        hook.pre_comm(step=self.S, level_number=0)
                    comm.Ibcast((np.array([1]), MPI.INT), root=self.S.status.slot).Wait()
                    for hook in self.hooks:
                        hook.post_comm(step=self.S, level_number=0, add_to_stats=True)
                    self.logger.debug(f'{self.S.status.slot} is done, broadcasting done')
                    self.S.status.done = True
                else:
                    for hook in self.hooks:
                        hook.pre_comm(step=self.S, level_number=0)
                    for hook in self.hooks:
                        hook.post_comm(step=self.S, level_number=0, add_to_stats=True)

    def pfasst(self, comm, num_procs):
        """
        Main function including the stages of SDC, MLSDC and PFASST (the "controller")

        For the workflow of this controller, check out one of our PFASST talks or the pySDC paper

        Args:
            comm: communicator
            num_procs (int): number of parallel processes
        """

        stage = self.S.status.stage

        self.logger.debug(stage + ' - process ' + str(self.S.status.slot))

        # Wait for interrupt, if iteration estimator is used
        if self.params.use_iteration_estimator and stage == 'SPREAD' and not self.S.status.last:
            done = np.empty(1)
            self.req_ibcast = comm.Ibcast((done, MPI.INT), root=comm.Get_size() - 1)

        # If interrupt is there, cleanup and finish
        if self.params.use_iteration_estimator and not self.S.status.last and self.req_ibcast.Test():
            self.logger.debug(f'{self.S.status.slot} is done..')
            self.S.status.done = True

            if not stage == 'IT_CHECK':
                self.logger.debug(f'Rewinding {self.S.status.slot} after {stage}..')
                self.S.levels[0].u[1:] = self.S.levels[0].uold[1:]

            for hook in self.hooks:
                hook.post_iteration(step=self.S, level_number=0)

            for req in self.req_send:
                if req is not None and req != MPI.REQUEST_NULL:
                    req.Cancel()
            if self.req_status is not None and self.req_status != MPI.REQUEST_NULL:
                self.req_status.Cancel()
            if self.req_diff is not None and self.req_diff != MPI.REQUEST_NULL:
                self.req_diff.Cancel()

            self.S.status.stage = 'DONE'
            for hook in self.hooks:
                hook.post_step(step=self.S, level_number=0)

        else:
            # Start cycling, if not interrupted
            switcher = {
                'SPREAD': self.spread,
                'PREDICT': self.predict,
                'IT_CHECK': self.it_check,
                'IT_FINE': self.it_fine,
                'IT_DOWN': self.it_down,
                'IT_COARSE': self.it_coarse,
                'IT_UP': self.it_up,
            }

            switcher.get(stage, self.default)(comm, num_procs)

    def spread(self, comm, num_procs):
        """
        Spreading phase
        """

        # first stage: spread values
        for hook in self.hooks:
            hook.pre_step(step=self.S, level_number=0)

        # call predictor from sweeper
        self.S.levels[0].sweep.predict()

        if self.params.use_iteration_estimator:
            # store pervious iterate to compute difference later on
            self.S.levels[0].uold[1:] = self.S.levels[0].u[1:]

        # update stage
        if len(self.S.levels) > 1:  # MLSDC or PFASST with predict
            self.S.status.stage = 'PREDICT'
        else:
            self.S.status.stage = 'IT_CHECK'

        for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
            C.post_spread_processing(self, self.S, comm=comm)

    def predict(self, comm, num_procs):
        """
        Predictor phase
        """

        for hook in self.hooks:
            hook.pre_predict(step=self.S, level_number=0)

        if self.params.predict_type is None:
            pass

        elif self.params.predict_type == 'fine_only':
            # do a fine sweep only
            self.S.levels[0].sweep.update_nodes()

        # elif self.params.predict_type == 'libpfasst_style':
        #
        #     # restrict to coarsest level
        #     for l in range(1, len(self.S.levels)):
        #         self.S.transfer(source=self.S.levels[l - 1], target=self.S.levels[l])
        #
        #     self.hooks.pre_comm(step=self.S, level_number=len(self.S.levels) - 1)
        #     if not self.S.status.first:
        #         self.logger.debug('recv data predict: process %s, stage %s, time, %s, source %s, tag %s' %
        #                           (self.S.status.slot, self.S.status.stage, self.S.time, self.S.prev,
        #                            self.S.status.iter))
        #         self.recv(target=self.S.levels[-1], source=self.S.prev, tag=self.S.status.iter, comm=comm)
        #     self.hooks.post_comm(step=self.S, level_number=len(self.S.levels) - 1)
        #
        #     # do the sweep with new values
        #     self.S.levels[-1].sweep.update_nodes()
        #     self.S.levels[-1].sweep.compute_end_point()
        #
        #     self.hooks.pre_comm(step=self.S, level_number=len(self.S.levels) - 1)
        #     if not self.S.status.last:
        #         self.logger.debug('send data predict: process %s, stage %s, time, %s, target %s, tag %s' %
        #                           (self.S.status.slot, self.S.status.stage, self.S.time, self.S.next,
        #                            self.S.status.iter))
        #         self.S.levels[-1].uend.isend(dest=self.S.next, tag=self.S.status.iter, comm=comm).Wait()
        #     self.hooks.post_comm(step=self.S, level_number=len(self.S.levels) - 1, add_to_stats=True)
        #
        #     # go back to fine level, sweeping
        #     for l in range(len(self.S.levels) - 1, 0, -1):
        #         # prolong values
        #         self.S.transfer(source=self.S.levels[l], target=self.S.levels[l - 1])
        #         # on middle levels: do sweep as usual
        #         if l - 1 > 0:
        #             self.S.levels[l - 1].sweep.update_nodes()
        #
        #     # end with a fine sweep
        #     self.S.levels[0].sweep.update_nodes()

        elif self.params.predict_type == 'pfasst_burnin':
            # restrict to coarsest level
            for l in range(1, len(self.S.levels)):
                self.S.transfer(source=self.S.levels[l - 1], target=self.S.levels[l])

            for p in range(self.S.status.slot + 1):
                if not p == 0:
                    self.recv_full(comm=comm, level=len(self.S.levels) - 1)
                    if self.S.status.force_done:
                        return None

                # do the sweep with new values
                self.S.levels[-1].sweep.update_nodes()
                self.S.levels[-1].sweep.compute_end_point()

                self.send_full(
                    comm=comm, blocking=True, level=len(self.S.levels) - 1, add_to_stats=(p == self.S.status.slot)
                )
                if self.S.status.force_done:
                    return None

            # interpolate back to finest level
            for l in range(len(self.S.levels) - 1, 0, -1):
                self.S.transfer(source=self.S.levels[l], target=self.S.levels[l - 1])

            self.send_full(comm=comm, level=0)
            if self.S.status.force_done:
                return None

            self.recv_full(comm=comm, level=0)
            if self.S.status.force_done:
                return None

            # end this with a fine sweep
            self.S.levels[0].sweep.update_nodes()

        elif self.params.predict_type == 'fmg':
            # TODO: implement FMG predictor
            raise NotImplementedError('FMG predictor is not yet implemented')

        else:
            raise ControllerError('Wrong predictor type, got %s' % self.params.predict_type)

        for hook in self.hooks:
            hook.post_predict(step=self.S, level_number=0)

        # update stage
        self.S.status.stage = 'IT_CHECK'

    def it_check(self, comm, num_procs):
        """
        Key routine to check for convergence/termination
        """

        # Update values to compute the residual
        self.send_full(comm=comm, level=0)
        if self.S.status.force_done:
            return None

        self.recv_full(comm=comm, level=0)
        if self.S.status.force_done:
            return None

        # compute the residual
        self.S.levels[0].sweep.compute_residual()

        if self.params.use_iteration_estimator:
            # TODO: replace with convergence controller
            self.check_iteration_estimate(comm=comm)

        if self.S.status.force_done:
            return None

        if self.S.status.iter > 0:
            for hook in self.hooks:
                hook.post_iteration(step=self.S, level_number=0)

        # decide if the step is done, needs to be restarted and other things convergence related
        for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
            C.post_iteration_processing(self, self.S, comm=comm)
            C.convergence_control(self, self.S, comm=comm)

        # if not ready, keep doing stuff
        if not self.S.status.done:
            # increment iteration count here (and only here)
            self.S.status.iter += 1

            for hook in self.hooks:
                hook.pre_iteration(step=self.S, level_number=0)
            for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
                C.pre_iteration_processing(self, self.S, comm=comm)

            if self.params.use_iteration_estimator:
                # store previous iterate to compute difference later on
                self.S.levels[0].uold[1:] = self.S.levels[0].u[1:]

            if len(self.S.levels) > 1:  # MLSDC or PFASST
                self.S.status.stage = 'IT_DOWN'
            else:
                if num_procs == 1 or self.params.mssdc_jac:  # SDC or parallel MSSDC (Jacobi-like)
                    self.S.status.stage = 'IT_FINE'
                else:
                    self.S.status.stage = 'IT_COARSE'  # serial MSSDC (Gauss-like)

        else:
            if not self.params.use_iteration_estimator:
                # Need to finish all pending isend requests. These will occur for the first active process, since
                # in the last iteration the wait statement will not be called ("send and forget")
                for req in self.req_send:
                    if req is not None:
                        req.Wait()
                if self.req_status is not None:
                    self.req_status.Wait()
                if self.req_diff is not None:
                    self.req_diff.Wait()
            else:
                for req in self.req_send:
                    if req is not None:
                        req.Cancel()
                if self.req_status is not None:
                    self.req_status.Cancel()
                if self.req_diff is not None:
                    self.req_diff.Cancel()

            for hook in self.hooks:
                hook.post_step(step=self.S, level_number=0)
            self.S.status.stage = 'DONE'

    def it_fine(self, comm, num_procs):
        """
        Fine sweeps
        """

        nsweeps = self.S.levels[0].params.nsweeps

        self.S.levels[0].status.sweep = 0

        # do fine sweep
        for k in range(nsweeps):
            self.S.levels[0].status.sweep += 1

            # send values forward
            self.send_full(comm=comm, level=0)
            if self.S.status.force_done:
                return None

            # recv values from previous
            self.recv_full(comm=comm, level=0, add_to_stats=(k == nsweeps - 1))
            if self.S.status.force_done:
                return None

            for hook in self.hooks:
                hook.pre_sweep(step=self.S, level_number=0)
            self.S.levels[0].sweep.update_nodes()
            for hook in self.hooks:
                hook.post_sweep(step=self.S, level_number=0)

        # update stage
        self.S.status.stage = 'IT_CHECK'

    def it_down(self, comm, num_procs):
        """
        Go down the hierarchy from finest to coarsest level
        """

        self.S.transfer(source=self.S.levels[0], target=self.S.levels[1])

        # sweep and send on middle levels (not on finest, not on coarsest, though)
        for l in range(1, len(self.S.levels) - 1):
            nsweeps = self.S.levels[l].params.nsweeps

            for _ in range(nsweeps):
                self.send_full(comm=comm, level=l)
                if self.S.status.force_done:
                    return None

                self.recv_full(comm=comm, level=l)
                if self.S.status.force_done:
                    return None

                for hook in self.hooks:
                    hook.pre_sweep(step=self.S, level_number=l)
                self.S.levels[l].sweep.update_nodes()
                self.S.levels[l].sweep.compute_residual()
                for hook in self.hooks:
                    hook.post_sweep(step=self.S, level_number=l)

            # transfer further down the hierarchy
            self.S.transfer(source=self.S.levels[l], target=self.S.levels[l + 1])

        # update stage
        self.S.status.stage = 'IT_COARSE'

    def it_coarse(self, comm, num_procs):
        """
        Coarse sweep
        """

        # receive from previous step (if not first)
        self.recv_full(comm=comm, level=len(self.S.levels) - 1)
        if self.S.status.force_done:
            return None

        # do the sweep
        for hook in self.hooks:
            hook.pre_sweep(step=self.S, level_number=len(self.S.levels) - 1)
        assert self.S.levels[-1].params.nsweeps == 1, (
            'ERROR: this controller can only work with one sweep on the coarse level, got %s'
            % self.S.levels[-1].params.nsweeps
        )
        self.S.levels[-1].sweep.update_nodes()
        self.S.levels[-1].sweep.compute_residual()
        for hook in self.hooks:
            hook.post_sweep(step=self.S, level_number=len(self.S.levels) - 1)
        self.S.levels[-1].sweep.compute_end_point()

        # send to next step
        self.send_full(comm=comm, blocking=True, level=len(self.S.levels) - 1, add_to_stats=True)
        if self.S.status.force_done:
            return None

        # update stage
        if len(self.S.levels) > 1:  # MLSDC or PFASST
            self.S.status.stage = 'IT_UP'
        else:
            self.S.status.stage = 'IT_CHECK'  # MSSDC

    def it_up(self, comm, num_procs):
        """
        Prolong corrections up to finest level (parallel)
        """

        # receive and sweep on middle levels (except for coarsest level)
        for l in range(len(self.S.levels) - 1, 0, -1):
            # prolong values
            self.S.transfer(source=self.S.levels[l], target=self.S.levels[l - 1])

            # on middle levels: do sweep as usual
            if l - 1 > 0:
                nsweeps = self.S.levels[l - 1].params.nsweeps

                for k in range(nsweeps):
                    self.send_full(comm, level=l - 1)
                    if self.S.status.force_done:
                        return None

                    self.recv_full(comm=comm, level=l - 1, add_to_stats=(k == nsweeps - 1))
                    if self.S.status.force_done:
                        return None

                    for hook in self.hooks:
                        hook.pre_sweep(step=self.S, level_number=l - 1)
                    self.S.levels[l - 1].sweep.update_nodes()
                    self.S.levels[l - 1].sweep.compute_residual()
                    for hook in self.hooks:
                        hook.post_sweep(step=self.S, level_number=l - 1)

        # update stage
        self.S.status.stage = 'IT_FINE'

    def default(self, num_procs):
        """
        Default routine to catch wrong status
        """
        raise ControllerError('Weird stage, got %s' % self.S.status.stage)
