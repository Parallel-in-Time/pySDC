import itertools
import copy as cp
import numpy as np
import dill

from pySDC.core.Controller import controller
from pySDC.core import Step as stepclass
from pySDC.core.Errors import ControllerError, CommunicationError
from pySDC.implementations.convergence_controller_classes.basic_restarting_nonMPI import BasicRestartingNonMPI


class controller_nonMPI(controller):
    """

    PFASST controller, running serialized version of PFASST in blocks (MG-style)

    """

    def __init__(self, num_procs, controller_params, description):
        """
        Initialization routine for PFASST controller

        Args:
           num_procs: number of parallel time steps (still serial, though), can be 1
           controller_params: parameter set for the controller and the steps
           description: all the parameters to set up the rest (levels, problems, transfer, ...)
        """

        if 'predict' in controller_params:
            raise ControllerError('predict flag is ignored, use predict_type instead')

        # call parent's initialization routine
        super(controller_nonMPI, self).__init__(controller_params, description)

        self.MS = [stepclass.step(description)]

        # try to initialize via dill.copy (much faster for many time-steps)
        try:
            for _ in range(num_procs - 1):
                self.MS.append(dill.copy(self.MS[0]))
        # if this fails (e.g. due to un-picklable data in the steps), initialize seperately
        except dill.PicklingError and TypeError:
            self.logger.warning('Need to initialize steps separately due to pickling error')
            for _ in range(num_procs - 1):
                self.MS.append(stepclass.step(description))

        if self.params.dump_setup:
            self.dump_setup(step=self.MS[0], controller_params=controller_params, description=description)

        if num_procs > 1 and len(self.MS[0].levels) > 1:
            for S in self.MS:
                for L in S.levels:
                    if not L.sweep.coll.right_is_node:
                        raise ControllerError("For PFASST to work, we assume uend^k = u_M^k")

        if all(len(S.levels) == len(self.MS[0].levels) for S in self.MS):
            self.nlevels = len(self.MS[0].levels)
        else:
            raise ControllerError('all steps need to have the same number of levels')

        if self.nlevels == 0:
            raise ControllerError('need at least one level')

        self.nsweeps = []
        for nl in range(self.nlevels):

            if all(S.levels[nl].params.nsweeps == self.MS[0].levels[nl].params.nsweeps for S in self.MS):
                self.nsweeps.append(self.MS[0].levels[nl].params.nsweeps)

        if self.nlevels > 1 and self.nsweeps[-1] > 1:
            raise ControllerError('this controller cannot do multiple sweeps on coarsest level')

        if self.nlevels == 1 and self.params.predict_type is not None:
            self.logger.warning(
                'you have specified a predictor type but only a single level.. ' 'predictor will be ignored'
            )

        self.add_convergence_controller(BasicRestartingNonMPI, description)

        for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
            C.reset_buffers_nonMPI(self)
            C.setup_status_variables(self)

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
        self.hooks.reset_stats()

        # initial ordering of the steps: 0,1,...,Np-1
        slots = list(range(num_procs))

        # initialize time variables of each step
        time = [t0 + sum(self.MS[j].dt for j in range(p)) for p in slots]

        # determine which steps are still active (time < Tend)
        active = [time[p] < Tend - 10 * np.finfo(float).eps for p in slots]

        if not any(active):
            raise ControllerError('Nothing to do, check t0, dt and Tend.')

        # compress slots according to active steps, i.e. remove all steps which have times above Tend
        active_slots = list(itertools.compress(slots, active))

        # initialize block of steps with u0
        self.restart_block(active_slots, time, u0)

        self.hooks.post_setup(step=None, level_number=None)

        # call pre-run hook
        for S in self.MS:
            self.hooks.pre_run(step=S, level_number=0)

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
                [C.prepare_next_block(self, S, len(active_slots), time, Tend) for S in self.MS]
                C.prepare_next_block_nonMPI(self, self.MS, active_slots, time, Tend)

            # setup the times of the steps for the next block
            for i in range(1, len(active_slots)):
                time[active_slots[i]] = time[active_slots[i] - 1] + self.MS[active_slots[i] - 1].dt

            # determine new set of active steps and compress slots accordingly
            active = [time[p] < Tend - 10 * np.finfo(float).eps for p in slots]
            active_slots = list(itertools.compress(slots, active))

            # restart active steps (reset all values and pass uend to u0)
            self.restart_block(active_slots, time, uend)

        # call post-run hook
        for S in self.MS:
            self.hooks.post_run(step=S, level_number=0)

        return uend, self.hooks.return_stats()

    def restart_block(self, active_slots, time, u0):
        """
        Helper routine to reset/restart block of (active) steps

        Args:
            active_slots: list of active steps
            time: list of new times
            u0: initial value to distribute across the steps

        """

        # loop over active slots (not directly, since we need the previous entry as well)
        for j in range(len(active_slots)):

            # get slot number
            p = active_slots[j]

            # store current slot number for diagnostics
            self.MS[p].status.slot = p
            # store link to previous step
            self.MS[p].prev = self.MS[active_slots[j - 1]]
            # resets step
            self.MS[p].reset_step()
            # determine whether I am the first and/or last in line
            self.MS[p].status.first = active_slots.index(p) == 0
            self.MS[p].status.last = active_slots.index(p) == len(active_slots) - 1
            # initialize step with u0
            self.MS[p].init_step(u0)
            # reset some values
            self.MS[p].status.done = False
            self.MS[p].status.prev_done = False
            self.MS[p].status.iter = 0
            self.MS[p].status.stage = 'SPREAD'
            self.MS[p].status.force_done = False
            self.MS[p].status.time_size = len(active_slots)
            self.MS[p].status.restart = False

            for l in self.MS[p].levels:
                l.tag = None
                l.status.sweep = 1

        for p in active_slots:
            for lvl in self.MS[p].levels:
                lvl.status.time = time[p]

    def send_full(self, S, level=None, add_to_stats=False):
        """
        Function to perform the send, including bookkeeping and logging

        Args:
            S: the current step
            level: the level number
            add_to_stats: a flag to end recording data in the hooks (defaults to False)
        """

        def send(source, tag):
            """
            Send function

            Args:
                source: level which has the new values
                tag: identifier for this message
            """
            # sending here means computing uend ("one-sided communication")
            source.sweep.compute_end_point()
            source.tag = cp.deepcopy(tag)

        self.hooks.pre_comm(step=S, level_number=level)
        if not S.status.last:
            self.logger.debug(
                'Process %2i provides data on level %2i with tag %s' % (S.status.slot, level, S.status.iter)
            )
            send(S.levels[level], tag=(level, S.status.iter, S.status.slot))
        self.hooks.post_comm(step=S, level_number=level, add_to_stats=add_to_stats)

    def recv_full(self, S, level=None, add_to_stats=False):
        """
        Function to perform the recv, including bookkeeping and logging

        Args:
            S: the current step
            level: the level number
            add_to_stats: a flag to end recording data in the hooks (defaults to False)
        """

        def recv(target, source, tag=None):
            """
            Receive function

            Args:
                target: level which will receive the values
                source: level which initiated the send
                tag: identifier to check if this message is really for me
            """

            if tag is not None and source.tag != tag:
                raise CommunicationError('source and target tag are not the same, got %s and %s' % (source.tag, tag))
            # simply do a deepcopy of the values uend to become the new u0 at the target
            target.u[0] = target.prob.dtype_u(source.uend)
            # re-evaluate f on left interval boundary
            target.f[0] = target.prob.eval_f(target.u[0], target.time)

        self.hooks.pre_comm(step=S, level_number=level)
        if not S.status.prev_done and not S.status.first:
            self.logger.debug(
                'Process %2i receives from %2i on level %2i with tag %s'
                % (S.status.slot, S.prev.status.slot, level, S.status.iter)
            )
            recv(S.levels[level], S.prev.levels[level], tag=(level, S.status.iter, S.prev.status.slot))
        self.hooks.post_comm(step=S, level_number=level, add_to_stats=add_to_stats)

    def pfasst(self, local_MS_active):
        """
        Main function including the stages of SDC, MLSDC and PFASST (the "controller")

        For the workflow of this controller, check out one of our PFASST talks or the pySDC paper

        This method changes self.MS directly by accessing active steps through local_MS_active. Nothing is returned.

        Args:
            local_MS_active (list): all active steps
        """

        # if all stages are the same (or DONE), continue, otherwise abort
        stages = [S.status.stage for S in local_MS_active if S.status.stage != 'DONE']
        if stages[1:] == stages[:-1]:
            stage = stages[0]
        else:
            raise ControllerError('not all stages are equal')

        self.logger.debug(stage)

        MS_running = [S for S in local_MS_active if S.status.stage != 'DONE']

        switcher = {
            'SPREAD': self.spread,
            'PREDICT': self.predict,
            'IT_CHECK': self.it_check,
            'IT_FINE': self.it_fine,
            'IT_DOWN': self.it_down,
            'IT_COARSE': self.it_coarse,
            'IT_UP': self.it_up,
        }

        switcher.get(stage, self.default)(MS_running)

        return all([S.status.done for S in local_MS_active])

    def spread(self, local_MS_running):
        """
        Spreading phase

        Args:
            local_MS_running (list): list of currently running steps
        """

        for S in local_MS_running:

            # first stage: spread values
            self.hooks.pre_step(step=S, level_number=0)

            # call predictor from sweeper
            S.levels[0].sweep.predict()

            # update stage
            if len(S.levels) > 1:  # MLSDC or PFASST with predict
                S.status.stage = 'PREDICT'
            else:
                S.status.stage = 'IT_CHECK'

            for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
                C.post_spread_processing(self, S)

    def predict(self, local_MS_running):
        """
        Predictor phase

        Args:
            local_MS_running (list): list of currently running steps
        """

        for S in local_MS_running:
            self.hooks.pre_predict(step=S, level_number=0)

        if self.params.predict_type is None:
            pass

        elif self.params.predict_type == 'fine_only':

            # do a fine sweep only
            for S in local_MS_running:
                S.levels[0].sweep.update_nodes()

        # elif self.params.predict_type == 'libpfasst_style':
        #
        #     # loop over all steps
        #     for S in local_MS_running:
        #
        #         # restrict to coarsest level
        #         for l in range(1, len(S.levels)):
        #             S.transfer(source=S.levels[l - 1], target=S.levels[l])
        #
        #     # run in serial on coarse level
        #     for S in local_MS_running:
        #
        #         self.hooks.pre_comm(step=S, level_number=len(S.levels) - 1)
        #         # receive from previous step (if not first)
        #         if not S.status.first:
        #             self.logger.debug('Process %2i receives from %2i on level %2i with tag %s -- PREDICT' %
        #                               (S.status.slot, S.prev.status.slot, len(S.levels) - 1, 0))
        #             self.recv(S.levels[-1], S.prev.levels[-1], tag=(len(S.levels), 0, S.prev.status.slot))
        #         self.hooks.post_comm(step=S, level_number=len(S.levels) - 1)
        #
        #         # do the coarse sweep
        #         S.levels[-1].sweep.update_nodes()
        #
        #         self.hooks.pre_comm(step=S, level_number=len(S.levels) - 1)
        #         # send to succ step
        #         if not S.status.last:
        #             self.logger.debug('Process %2i provides data on level %2i with tag %s -- PREDICT'
        #                               % (S.status.slot, len(S.levels) - 1, 0))
        #             self.send(S.levels[-1], tag=(len(S.levels), 0, S.status.slot))
        #         self.hooks.post_comm(step=S, level_number=len(S.levels) - 1, add_to_stats=True)
        #
        #     # go back to fine level, sweeping
        #     for l in range(self.nlevels - 1, 0, -1):
        #
        #         for S in local_MS_running:
        #             # prolong values
        #             S.transfer(source=S.levels[l], target=S.levels[l - 1])
        #
        #             if l - 1 > 0:
        #                 S.levels[l - 1].sweep.update_nodes()
        #
        #     # end with a fine sweep
        #     for S in local_MS_running:
        #         S.levels[0].sweep.update_nodes()

        elif self.params.predict_type == 'pfasst_burnin':

            # loop over all steps
            for S in local_MS_running:

                # restrict to coarsest level
                for l in range(1, len(S.levels)):
                    S.transfer(source=S.levels[l - 1], target=S.levels[l])

            # loop over all steps
            for q in range(len(local_MS_running)):

                # loop over last steps: [1,2,3,4], [2,3,4], [3,4], [4]
                for p in range(q, len(local_MS_running)):
                    S = local_MS_running[p]

                    # do the sweep with new values
                    S.levels[-1].sweep.update_nodes()

                    # send updated values on coarsest level
                    self.send_full(S, level=len(S.levels) - 1)

                # loop over last steps: [2,3,4], [3,4], [4]
                for p in range(q + 1, len(local_MS_running)):
                    S = local_MS_running[p]
                    # receive values sent during previous sweep
                    self.recv_full(S, level=len(S.levels) - 1, add_to_stats=(p == len(local_MS_running) - 1))

            # loop over all steps
            for S in local_MS_running:

                # interpolate back to finest level
                for l in range(len(S.levels) - 1, 0, -1):
                    S.transfer(source=S.levels[l], target=S.levels[l - 1])

                # send updated values forward
                self.send_full(S, level=0)
                # receive values
                self.recv_full(S, level=0)

            # end this with a fine sweep
            for S in local_MS_running:
                S.levels[0].sweep.update_nodes()

        elif self.params.predict_type == 'fmg':
            # TODO: implement FMG predictor
            raise NotImplementedError('FMG predictor is not yet implemented')

        else:
            raise ControllerError('Wrong predictor type, got %s' % self.params.predict_type)

        for S in local_MS_running:
            self.hooks.post_predict(step=S, level_number=0)

        for S in local_MS_running:
            # update stage
            S.status.stage = 'IT_CHECK'

    def it_check(self, local_MS_running):
        """
        Key routine to check for convergence/termination

        Args:
            local_MS_running (list): list of currently running steps
        """

        for S in local_MS_running:

            # send updated values forward
            self.send_full(S, level=0)
            # receive values
            self.recv_full(S, level=0)
            # compute current residual
            S.levels[0].sweep.compute_residual()

        for S in local_MS_running:

            if S.status.iter > 0:
                self.hooks.post_iteration(step=S, level_number=0)

            # decide if the step is done, needs to be restarted and other things convergence related
            for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
                C.post_iteration_processing(self, S)
                C.convergence_control(self, S)

        for S in local_MS_running:
            if not S.status.first:
                self.hooks.pre_comm(step=S, level_number=0)
                S.status.prev_done = S.prev.status.done  # "communicate"
                self.hooks.post_comm(step=S, level_number=0, add_to_stats=True)
                S.status.done = S.status.done and S.status.prev_done

            if self.params.all_to_done:
                self.hooks.pre_comm(step=S, level_number=0)
                S.status.done = all([T.status.done for T in local_MS_running])
                self.hooks.post_comm(step=S, level_number=0, add_to_stats=True)

            if not S.status.done:
                # increment iteration count here (and only here)
                S.status.iter += 1
                self.hooks.pre_iteration(step=S, level_number=0)

                if len(S.levels) > 1:  # MLSDC or PFASST
                    S.status.stage = 'IT_DOWN'
                else:  # SDC or MSSDC
                    if len(local_MS_running) == 1 or self.params.mssdc_jac:  # SDC or parallel MSSDC (Jacobi-like)
                        S.status.stage = 'IT_FINE'
                    else:
                        S.status.stage = 'IT_COARSE'  # serial MSSDC (Gauss-like)
            else:
                S.levels[0].sweep.compute_end_point()
                self.hooks.post_step(step=S, level_number=0)
                S.status.stage = 'DONE'

        for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
            C.reset_buffers_nonMPI(self)

    def it_fine(self, local_MS_running):
        """
        Fine sweeps

        Args:
            local_MS_running (list): list of currently running steps
        """

        for S in local_MS_running:
            S.levels[0].status.sweep = 0

        for k in range(self.nsweeps[0]):

            for S in local_MS_running:
                S.levels[0].status.sweep += 1

            for S in local_MS_running:
                # send updated values forward
                self.send_full(S, level=0)
                # receive values
                self.recv_full(S, level=0, add_to_stats=(k == self.nsweeps[0] - 1))

            for S in local_MS_running:
                # standard sweep workflow: update nodes, compute residual, log progress
                self.hooks.pre_sweep(step=S, level_number=0)
                S.levels[0].sweep.update_nodes()
                S.levels[0].sweep.compute_residual()
                self.hooks.post_sweep(step=S, level_number=0)

        for S in local_MS_running:
            # update stage
            S.status.stage = 'IT_CHECK'

    def it_down(self, local_MS_running):
        """
        Go down the hierarchy from finest to coarsest level

        Args:
            local_MS_running (list): list of currently running steps
        """

        for S in local_MS_running:
            S.transfer(source=S.levels[0], target=S.levels[1])

        for l in range(1, self.nlevels - 1):

            # sweep on middle levels (not on finest, not on coarsest, though)

            for _ in range(self.nsweeps[l]):

                for S in local_MS_running:

                    # send updated values forward
                    self.send_full(S, level=l)
                    # receive values
                    self.recv_full(S, level=l)

                for S in local_MS_running:
                    self.hooks.pre_sweep(step=S, level_number=l)
                    S.levels[l].sweep.update_nodes()
                    S.levels[l].sweep.compute_residual()
                    self.hooks.post_sweep(step=S, level_number=l)

            for S in local_MS_running:
                # transfer further down the hierarchy
                S.transfer(source=S.levels[l], target=S.levels[l + 1])

        for S in local_MS_running:
            # update stage
            S.status.stage = 'IT_COARSE'

    def it_coarse(self, local_MS_running):
        """
        Coarse sweep

        Args:
            local_MS_running (list): list of currently running steps
        """

        for S in local_MS_running:

            # receive from previous step (if not first)
            self.recv_full(S, level=len(S.levels) - 1)

            # do the sweep
            self.hooks.pre_sweep(step=S, level_number=len(S.levels) - 1)
            S.levels[-1].sweep.update_nodes()
            S.levels[-1].sweep.compute_residual()
            self.hooks.post_sweep(step=S, level_number=len(S.levels) - 1)

            # send to succ step
            self.send_full(S, level=len(S.levels) - 1, add_to_stats=True)

            # update stage
            if len(S.levels) > 1:  # MLSDC or PFASST
                S.status.stage = 'IT_UP'
            else:  # MSSDC
                S.status.stage = 'IT_CHECK'

    def it_up(self, local_MS_running):
        """
        Prolong corrections up to finest level (parallel)

        Args:
            local_MS_running (list): list of currently running steps
        """

        for l in range(self.nlevels - 1, 0, -1):

            for S in local_MS_running:
                # prolong values
                S.transfer(source=S.levels[l], target=S.levels[l - 1])

            # on middle levels: do communication and sweep as usual
            if l - 1 > 0:

                for k in range(self.nsweeps[l - 1]):

                    for S in local_MS_running:

                        # send updated values forward
                        self.send_full(S, level=l - 1)
                        # receive values
                        self.recv_full(S, level=l - 1, add_to_stats=(k == self.nsweeps[l - 1] - 1))

                    for S in local_MS_running:
                        self.hooks.pre_sweep(step=S, level_number=l - 1)
                        S.levels[l - 1].sweep.update_nodes()
                        S.levels[l - 1].sweep.compute_residual()
                        self.hooks.post_sweep(step=S, level_number=l - 1)

        for S in local_MS_running:
            # update stage
            S.status.stage = 'IT_FINE'

    def default(self, local_MS_running):
        """
        Default routine to catch wrong status

        Args:
            local_MS_running (list): list of currently running steps
        """
        raise ControllerError('Unknown stage, got %s' % local_MS_running[0].status.stage)  # TODO
