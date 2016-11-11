import numpy as np

from pySDC_core.Controller import controller
from pySDC_core.Step import step
from pySDC_core.Errors import ControllerError


class allinclusive_classic_MPI(controller):
    """

    PFASST controller, running parallel version of PFASST in classical style

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
        super(allinclusive_classic_MPI, self).__init__(controller_params)

        # create single step per processor
        self.S = step(description)

        # pass communicator for future use
        self.comm = comm
        # add request handle container for isend
        self.req_send = []
        # add request handler for status send
        self.req_status = None

        num_procs = self.comm.Get_size()
        rank = self.comm.Get_rank()

        if self.params.dump_setup and rank == 0:
            self.dump_setup(step=self.S, controller_params=controller_params, description=description)

        num_levels = len(self.S.levels)

        if num_procs > 1 and num_levels > 1:
            for L in self.S.levels:
                if not L.sweep.coll.right_is_node or L.sweep.params.do_coll_update:
                    raise ControllerError("For PFASST to work, we assume uend^k = u_M^k in this controller")

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
        self.hooks.reset_stats()

        # find active processes and put into new communicator
        rank = self.comm.Get_rank()
        all_dt = self.comm.allgather(self.S.dt)
        time = t0 + sum(all_dt[0:rank])
        active = time < Tend - 10 * np.finfo(float).eps
        comm_active = self.comm.Split(active)
        rank = comm_active.Get_rank()
        num_procs = comm_active.Get_size()
        self.S.status.slot = rank

        # initialize block of steps with u0
        self.restart_block(num_procs, time, u0)
        uend = u0

        # call pre-run hook
        self.hooks.pre_run(step=self.S, level_number=0)

        # while any process still active...
        while active:

            while not self.S.status.done:
                self.pfasst(comm_active, num_procs)

            time += self.S.dt

            # broadcast uend, set new times and fine active processes
            tend = comm_active.bcast(time, root=num_procs - 1)
            uend = comm_active.bcast(self.S.levels[0].uend, root=num_procs - 1)
            all_dt = comm_active.allgather(self.S.dt)
            time = tend + sum(all_dt[0:rank])
            active = time < Tend - 10 * np.finfo(float).eps
            comm_active = comm_active.Split(active)
            rank = comm_active.Get_rank()
            num_procs = comm_active.Get_size()
            self.S.status.slot = rank

            # initialize block of steps with u0
            self.restart_block(num_procs, time, uend)

        # call post-run hook
        self.hooks.post_run(step=self.S, level_number=0)

        comm_active.Free()

        return uend, self.hooks.return_stats()

    def restart_block(self, size, time, u0):
        """
        Helper routine to reset/restart block of (active) steps

        Args:
            size: number of active time steps
            time: current time
            u0: initial value to distribute across the steps

        Returns:
            block of (all) steps
        """

        # store link to previous step
        self.S.prev = self.S.status.slot - 1
        self.S.next = self.S.status.slot + 1
        # resets step
        self.S.reset_step()
        # determine whether I am the first and/or last in line
        self.S.status.first = self.S.prev == -1
        self.S.status.last = self.S.next == size
        # intialize step with u0
        self.S.init_step(u0)
        # reset some values
        self.S.status.done = False
        self.S.status.iter = 1
        self.S.status.stage = 'SPREAD'
        for l in self.S.levels:
            l.tag = None
        self.req_status = None
        self.req_send = []
        self.S.status.prev_done = False

        for lvl in self.S.levels:
            lvl.status.time = time

    @staticmethod
    def recv(target, source, tag, comm):
        """
        Receive function

        Args:
            target: process/level which will receive the values
            source: process/level which initiated the send
            tag: identifier to check if this message is really for me
            comm: communicator
        """
        # do a blocking receive and re-compute f(u0)
        target.u[0] = comm.recv(source=source, tag=tag)
        target.f[0] = target.prob.eval_f(target.u[0], target.time)

    @staticmethod
    def send(source, target, tag, comm):
        """
        Send function

        Args:
            source: process/level which has the new values
            target: process/level which will receive the values
            tag: identifier for this message
            comm: communicator
        """
        # do a blocking send
        comm.send(source.uend, dest=target, tag=tag)

    def predictor(self, comm):
        """
        Predictor function, extracted from the stepwise implementation (will be also used by matrix sweppers)

        Args:
            comm: communicator
        """

        # restrict to coarsest level
        for l in range(1, len(self.S.levels)):
            self.S.transfer(source=self.S.levels[l - 1], target=self.S.levels[l])

        for p in range(self.S.status.slot + 1):

            if not p == 0 and not self.S.status.first:
                self.logger.debug('recv data predict: process %s, stage %s, time, %s, source %s, tag %s, phase %s' %
                                  (self.S.status.slot, self.S.status.stage, self.S.time, self.S.prev,
                                   self.S.status.iter, p))
                self.recv(target=self.S.levels[-1], source=self.S.prev, tag=self.S.status.iter, comm=comm)

            # do the sweep with new values
            self.S.levels[-1].sweep.update_nodes()
            self.S.levels[-1].sweep.compute_end_point()

            if not self.S.status.last:
                self.logger.debug('send data predict: process %s, stage %s, time, %s, target %s, tag %s, phase %s' %
                                  (self.S.status.slot, self.S.status.stage, self.S.time, self.S.next,
                                   self.S.status.iter, p))
                self.send(source=self.S.levels[-1], target=self.S.next, tag=self.S.status.iter, comm=comm)

        # interpolate back to finest level
        for l in range(len(self.S.levels) - 1, 0, -1):
            self.S.transfer(source=self.S.levels[l], target=self.S.levels[l - 1])

    def pfasst(self, comm, num_procs):
        """
        Main function including the stages of SDC, MLSDC and PFASST (the "controller")

        For the workflow of this controller, check out one of our PFASST talks

        Args:
            comm: communicator
            num_procs: number of active processors
        """

        stage = self.S.status.stage

        self.logger.debug(stage)

        if stage == 'SPREAD':
            # (potentially) serial spreading phase

            # first stage: spread values
            self.hooks.pre_step(step=self.S, level_number=0)

            # call predictor from sweeper
            self.S.levels[0].sweep.predict()

            # update stage
            if len(self.S.levels) > 1 and self.params.predict:  # MLSDC or PFASST with predict
                self.S.status.stage = 'PREDICT'
            elif len(self.S.levels) > 1:  # MLSDC or PFASST without predict
                self.hooks.pre_iteration(step=self.S, level_number=0)
                self.S.status.stage = 'IT_FINE'
            elif num_procs > 1:  # MSSDC
                self.hooks.pre_iteration(step=self.S, level_number=0)
                self.S.status.stage = 'IT_COARSE'
            elif num_procs == 1:  # SDC
                self.hooks.pre_iteration(step=self.S, level_number=0)
                self.S.status.stage = 'IT_FINE'
            else:
                raise ControllerError("Don't know what to do after spread, aborting")

        elif stage == 'PREDICT':

            # call predictor (serial)

            self.predictor(comm)

            # update stage
            self.hooks.pre_iteration(step=self.S, level_number=0)
            self.S.status.stage = 'IT_FINE'

        elif stage == 'IT_FINE':

            # do fine sweep

            # standard sweep workflow: update nodes, compute residual, log progress
            self.hooks.pre_sweep(step=self.S, level_number=0)
            self.S.levels[0].sweep.update_nodes()
            self.S.levels[0].sweep.compute_residual()
            self.hooks.post_sweep(step=self.S, level_number=0)

            # wait for pending sends before computing uend, if any
            if len(self.req_send) > 0 and not self.S.status.last and self.params.fine_comm:
                self.req_send[0].wait()

            self.S.levels[0].sweep.compute_end_point()

            if not self.S.status.last and self.params.fine_comm:
                self.logger.debug('isend data: process %s, stage %s, time %s, target %s, tag %s, iter %s' %
                                  (self.S.status.slot, self.S.status.stage, self.S.time, self.S.next,
                                   0, self.S.status.iter))
                self.req_send.append(comm.isend(self.S.levels[0].uend, dest=self.S.next, tag=0))

            # update stage
            self.S.status.stage = 'IT_CHECK'

        elif stage == 'IT_CHECK':

            # check whether to stop iterating (parallel)

            self.hooks.post_iteration(step=self.S, level_number=0)

            # check if an open request of the status send is pending
            if self.req_status is not None:
                self.req_status.wait()

            # check for convergence or abort
            self.S.status.done = self.check_convergence(self.S)

            # send status forward
            if not self.S.status.last:
                self.logger.debug('isend status: status %s, process %s, time %s, target %s, tag %s, iter %s' %
                                  (self.S.status.done, self.S.status.slot, self.S.time, self.S.next,
                                   99, self.S.status.iter))
                self.req_status = comm.isend(self.S.status.done, dest=self.S.next, tag=99)

            # recv status
            if not self.S.status.first and not self.S.status.prev_done:
                self.S.status.prev_done = comm.recv(source=self.S.prev, tag=99)
                self.logger.debug('recv status: status %s, process %s, time %s, target %s, tag %s, iter %s' %
                                  (self.S.status.prev_done, self.S.status.slot, self.S.time, self.S.next,
                                   99, self.S.status.iter))
                self.S.status.done = self.S.status.done and self.S.status.prev_done

            # if I'm not done or the guy left of me is not done, keep doing stuff
            if not self.S.status.done:
                # increment iteration count here (and only here)
                self.S.status.iter += 1
                self.hooks.pre_iteration(step=self.S, level_number=0)
                # multi-level or single-level?
                if len(self.S.levels) > 1:  # MLSDC or PFASST
                    self.S.status.stage = 'IT_UP'
                elif num_procs > 1:  # MSSDC
                    self.S.status.stage = 'IT_COARSE_RECV'
                elif num_procs == 1:  # SDC
                    self.S.status.stage = 'IT_FINE'
                else:
                    raise ControllerError("Weird stage in IT_CHECK")

            else:
                self.S.levels[0].sweep.compute_end_point()
                self.hooks.post_step(step=self.S, level_number=0)
                self.S.status.stage = 'DONE'

        elif stage == 'IT_UP':

            # go up the hierarchy from finest to coarsest level (parallel)

            self.S.transfer(source=self.S.levels[0], target=self.S.levels[1])

            # sweep and send on middle levels (not on finest, not on coarsest, though)
            for l in range(1, len(self.S.levels) - 1):
                self.hooks.pre_sweep(step=self.S, level_number=l)
                self.S.levels[l].sweep.update_nodes()
                self.S.levels[l].sweep.compute_residual()
                self.hooks.post_sweep(step=self.S, level_number=l)

                # wait for pending sends before computing uend, if any
                if len(self.req_send) > l and not self.S.status.last and self.params.fine_comm:
                    self.req_send[l].wait()

                self.S.levels[l].sweep.compute_end_point()

                if not self.S.status.last and self.params.fine_comm:
                    self.logger.debug('isend data: process %s, stage %s, time %s, target %s, tag %s, iter %s' %
                                      (self.S.status.slot, self.S.status.stage, self.S.time, self.S.next,
                                       l, self.S.status.iter))
                    self.req_send.append(comm.isend(self.S.levels[l].uend, dest=self.S.next, tag=l))

                # transfer further up the hierarchy
                self.S.transfer(source=self.S.levels[l], target=self.S.levels[l + 1])

            # update stage
            self.S.status.stage = 'IT_COARSE_RECV'

        elif stage == 'IT_COARSE_RECV':

            # receive from previous step (if not first)
            if not self.S.status.first and not self.S.status.prev_done:
                self.logger.debug('recv data: process %s, stage %s, time %s, source %s, tag %s, iter %s' %
                                  (self.S.status.slot, self.S.status.stage, self.S.time, self.S.prev,
                                   len(self.S.levels) - 1, self.S.status.iter))
                self.recv(target=self.S.levels[-1], source=self.S.prev, tag=len(self.S.levels) - 1, comm=comm)

            # update stage
            self.S.status.stage = 'IT_COARSE'

        elif stage == 'IT_COARSE':

            # sweeps on coarsest level (serial/blocking)

            # do the sweep
            self.hooks.pre_sweep(step=self.S, level_number=len(self.S.levels) - 1)
            self.S.levels[-1].sweep.update_nodes()
            self.S.levels[-1].sweep.compute_residual()
            self.hooks.post_sweep(step=self.S, level_number=len(self.S.levels) - 1)
            self.S.levels[-1].sweep.compute_end_point()

            # send to next step
            if not self.S.status.last:
                self.logger.debug('isend data: process %s, stage %s, time %s, target %s, tag %s, iter %s' %
                                  (self.S.status.slot, self.S.status.stage, self.S.time, self.S.next,
                                   len(self.S.levels) - 1, self.S.status.iter))
                self.send(source=self.S.levels[-1], target=self.S.next, tag=len(self.S.levels) - 1, comm=comm)

            # update stage
            if len(self.S.levels) > 1:  # MLSDC or PFASST
                self.S.status.stage = 'IT_DOWN'
            else:  # MSSDC
                self.S.status.stage = 'IT_CHECK'

        elif stage == 'IT_DOWN':

            # prolong corrections down to finest level (parallel)

            # receive and sweep on middle levels (except for coarsest level)
            for l in range(len(self.S.levels) - 1, 0, -1):

                if not self.S.status.first and self.params.fine_comm and not self.S.status.prev_done:
                    self.logger.debug('recv data: process %s, stage %s, time %s, source %s, tag %s, iter %s' %
                                      (self.S.status.slot, self.S.status.stage, self.S.time, self.S.prev,
                                       l - 1, self.S.status.iter))
                    self.recv(target=self.S.levels[l - 1], source=self.S.prev, tag=l - 1, comm=comm)

                # prolong values
                self.S.transfer(source=self.S.levels[l], target=self.S.levels[l - 1])

                # on middle levels: do sweep as usual
                if l - 1 > 0:
                    self.hooks.pre_sweep(step=self.S, level_number=l - 1)
                    self.S.levels[l - 1].sweep.update_nodes()
                    self.S.levels[l - 1].sweep.compute_residual()
                    self.hooks.post_sweep(step=self.S, level_number=l - 1)

            # update stage
            self.S.status.stage = 'IT_FINE'

        else:

            raise ControllerError('Weird stage, got %s' % self.S.status.stage)
