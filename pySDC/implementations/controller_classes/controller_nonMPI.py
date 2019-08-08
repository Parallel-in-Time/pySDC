import itertools
import copy as cp
import numpy as np
import dill

from pySDC.core.Controller import controller
from pySDC.core import Step as stepclass
from pySDC.core.Errors import ControllerError, CommunicationError


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
        super(controller_nonMPI, self).__init__(controller_params)

        self.MS = [stepclass.step(description)]

        # try to initialize via dill.copy (much faster for many time-steps)
        try:
            for p in range(num_procs - 1):
                self.MS.append(dill.copy(self.MS[0]))
        # if this fails (e.g. due to un-picklable data in the steps), initialize seperately
        except dill.PicklingError and TypeError:
            self.logger.warning('Need to initialize steps separately due to pickling error')
            for p in range(num_procs - 1):
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
            self.logger.warning('you have specified a predictor type but only a single level.. '
                                'predictor will be ignored')

    @staticmethod
    def check_iteration_estimator(MS):
        """
        Method to check the iteration estimator

        Args:
            MS (list): list of currently active steps
        """
        diff_new = 0.0
        Kest_loc = 99

        # go through active steps and compute difference, Ltilde, Kest up to this step
        for S in MS:
            L = S.levels[0]

            for m in range(1, L.sweep.coll.num_nodes + 1):
                diff_new = max(diff_new, abs(L.uold[m] - L.u[m]))

            if S.status.iter == 1:
                S.status.diff_old_loc = diff_new
                S.status.diff_first_loc = diff_new
            elif S.status.iter > 1:
                Ltilde_loc = min(diff_new / S.status.diff_old_loc, 0.9)
                S.status.diff_old_loc = diff_new
                alpha = 1 / (1 - Ltilde_loc) * S.status.diff_first_loc
                Kest_loc = np.log(S.params.err_tol / alpha) / np.log(Ltilde_loc)
                print(f'LOCAL: {L.time}, {S.status.iter}: {int(np.ceil(Kest_loc))}, {Ltilde_loc}, {L.status.residual}')
                # You should not stop prematurely on earlier steps, since later steps may need more accuracy to reach
                # the tolerance themselves. The final Kest_loc is the one that counts.
                # if np.ceil(Kest_loc) <= S.status.iter:
                #     S.status.force_done = True

        # set global Kest as last local one, force stop if done
        for S in MS:
            if S.status.iter > 1:
                Kest_glob = Kest_loc
                if np.ceil(Kest_glob) <= S.status.iter:
                    S.status.force_done = True

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
        slots = [p for p in range(num_procs)]

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

            MS_active = []
            for p in active_slots:
                MS_active.append(self.MS[p])

            while not all([MS_active[p].status.done for p in range(len(MS_active))]):
                MS_active = self.pfasst(MS_active)

            for p in range(len(MS_active)):
                self.MS[active_slots[p]] = MS_active[p]

            # uend is uend of the last active step in the list
            uend = self.MS[active_slots[-1]].levels[0].uend

            for p in active_slots:
                time[p] += num_procs * self.MS[p].dt

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

            for l in self.MS[p].levels:
                l.tag = None
                l.status.sweep = 1

        for p in active_slots:
            for lvl in self.MS[p].levels:
                lvl.status.time = time[p]

    @staticmethod
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

    @staticmethod
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

    def predictor(self, MS):
        """
        Predictor function, extracted from the stepwise implementation (will be also used by matrix sweppers)

        Args:
            MS: all active steps

        Returns:
            all active steps
        """

        if self.params.predict_type is None:
            pass

        elif self.params.predict_type == 'fine_only':

            # do a fine sweep only
            for S in MS:
                S.levels[0].sweep.update_nodes()

        elif self.params.predict_type == 'libpfasst_style':

            # loop over all steps
            for S in MS:

                # restrict to coarsest level
                for l in range(1, len(S.levels)):
                    S.transfer(source=S.levels[l - 1], target=S.levels[l])

            # run in serial on coarse level
            for S in MS:

                self.hooks.pre_comm(step=S, level_number=len(S.levels) - 1)
                # receive from previous step (if not first)
                if not S.status.first:
                    self.logger.debug('Process %2i receives from %2i on level %2i with tag %s -- PREDICT' %
                                      (S.status.slot, S.prev.status.slot, len(S.levels) - 1, 0))
                    self.recv(S.levels[-1], S.prev.levels[-1], tag=(len(S.levels), 0, S.prev.status.slot))
                self.hooks.post_comm(step=S, level_number=len(S.levels) - 1)

                # do the coarse sweep
                S.levels[-1].sweep.update_nodes()

                self.hooks.pre_comm(step=S, level_number=len(S.levels) - 1)
                # send to succ step
                if not S.status.last:
                    self.logger.debug('Process %2i provides data on level %2i with tag %s -- PREDICT'
                                      % (S.status.slot, len(S.levels) - 1, 0))
                    self.send(S.levels[-1], tag=(len(S.levels), 0, S.status.slot))
                self.hooks.post_comm(step=S, level_number=len(S.levels) - 1, add_to_stats=True)

            # go back to fine level, sweeping
            for l in range(self.nlevels - 1, 0, -1):

                for S in MS:
                    # prolong values
                    S.transfer(source=S.levels[l], target=S.levels[l - 1])

                    if l - 1 > 0:
                        S.levels[l - 1].sweep.update_nodes()

            # end with a fine sweep
            for S in MS:
                S.levels[0].sweep.update_nodes()

        elif self.params.predict_type == 'pfasst_burnin':

            # loop over all steps
            for S in MS:

                # restrict to coarsest level
                for l in range(1, len(S.levels)):
                    S.transfer(source=S.levels[l - 1], target=S.levels[l])

            # loop over all steps
            for q in range(len(MS)):

                # loop over last steps: [1,2,3,4], [2,3,4], [3,4], [4]
                for p in range(q, len(MS)):
                    S = MS[p]

                    # do the sweep with new values
                    S.levels[-1].sweep.update_nodes()

                    self.hooks.pre_comm(step=S, level_number=len(S.levels) - 1)
                    # send updated values on coarsest level
                    self.logger.debug('Process %2i provides data on level %2i with tag %s -- PREDICT'
                                      % (S.status.slot, len(S.levels) - 1, 0))
                    self.send(S.levels[-1], tag=(len(S.levels), 0, S.status.slot))
                    self.hooks.post_comm(step=S, level_number=len(S.levels) - 1)

                # loop over last steps: [2,3,4], [3,4], [4]
                for p in range(q + 1, len(MS)):
                    S = MS[p]
                    # receive values sent during previous sweep
                    self.hooks.pre_comm(step=S, level_number=len(S.levels) - 1)
                    self.logger.debug('Process %2i receives from %2i on level %2i with tag %s -- PREDICT' %
                                      (S.status.slot, S.prev.status.slot, len(S.levels) - 1, 0))
                    self.recv(S.levels[-1], S.prev.levels[-1], tag=(len(S.levels), 0, S.prev.status.slot))
                    self.hooks.post_comm(step=S, level_number=len(S.levels) - 1, add_to_stats=(p == len(MS) - 1))

            # loop over all steps
            for S in MS:

                # interpolate back to finest level
                for l in range(len(S.levels) - 1, 0, -1):
                    S.transfer(source=S.levels[l], target=S.levels[l - 1])

            # end this with a fine sweep
            for S in MS:
                S.levels[0].sweep.update_nodes()

        elif self.params.predict_type == 'fmg':
            # TODO: implement FMG predictor
            raise NotImplementedError('FMG predictor is not yet implemented')

        else:
            raise ControllerError('Wrong predictor type, got %s' % self.params.predict_type)

        return MS

    def pfasst(self, MS):
        """
        Main function including the stages of SDC, MLSDC and PFASST (the "controller")

        For the workflow of this controller, check out one of our PFASST talks

        Args:
            MS: all active steps

        Returns:
            all active steps
        """

        # if all stages are the same (or DONE), continue, otherwise abort
        stages = [S.status.stage for S in MS if S.status.stage is not 'DONE']
        if stages[1:] == stages[:-1]:
            stage = stages[0]
        else:
            raise ControllerError('not all stages are equal')

        self.logger.debug(stage)

        MS_active = [S for S in MS if S.status.stage is not 'DONE']

        if stage == 'SPREAD':
            # (potentially) serial spreading phase
            for S in MS_active:

                # first stage: spread values
                self.hooks.pre_step(step=S, level_number=0)

                # call predictor from sweeper
                S.levels[0].sweep.predict()

                # store pervious iterate to compute difference later on
                S.levels[0].uold[:] = S.levels[0].u[:]

                # update stage
                if len(S.levels) > 1:  # MLSDC or PFASST with predict
                    S.status.stage = 'PREDICT'
                else:
                    S.status.stage = 'IT_CHECK'

            return MS

        elif stage == 'PREDICT':
            # call predictor (serial)

            for S in MS_active:
                self.hooks.pre_predict(step=S, level_number=0)

            MS = self.predictor(MS_active)

            for S in MS_active:
                self.hooks.post_predict(step=S, level_number=0)

            for S in MS_active:
                # update stage
                S.status.stage = 'IT_CHECK'

            return MS

        elif stage == 'IT_CHECK':

            # check whether to stop iterating (parallel)

            for S in MS_active:

                # send updated values forward
                self.hooks.pre_comm(step=S, level_number=0)
                if self.params.fine_comm and not S.status.last:
                    self.logger.debug('Process %2i provides data on level %2i with tag %s'
                                      % (S.status.slot, 0, S.status.iter))
                    self.send(S.levels[0], tag=(0, S.status.iter, S.status.slot))

                # receive values
                if self.params.fine_comm and not S.status.prev_done and not S.status.first:
                    self.logger.debug('Process %2i receives from %2i on level %2i with tag %s' %
                                      (S.status.slot, S.prev.status.slot, 0, S.status.iter))
                    self.recv(S.levels[0], S.prev.levels[0], tag=(0, S.status.iter, S.prev.status.slot))
                self.hooks.post_comm(step=S, level_number=0)

                S.levels[0].sweep.compute_residual()

            if self.params.use_iteration_estimator:
                self.check_iteration_estimator(MS_active)

            for S in MS_active:

                S.status.done = self.check_convergence(S)

                if S.status.iter > 0:
                    self.hooks.post_iteration(step=S, level_number=0)

            for S in MS_active:
                if not S.status.first:
                    self.hooks.pre_comm(step=S, level_number=0)
                    S.status.prev_done = S.prev.status.done  # "communicate"
                    self.hooks.post_comm(step=S, level_number=0, add_to_stats=True)
                    S.status.done = S.status.done and S.status.prev_done

                if self.params.all_to_done:
                    self.hooks.pre_comm(step=S, level_number=0)
                    S.status.done = all([T.status.done for T in MS])
                    self.hooks.post_comm(step=S, level_number=0, add_to_stats=True)

                if not S.status.done:
                    # increment iteration count here (and only here)
                    S.status.iter += 1
                    self.hooks.pre_iteration(step=S, level_number=0)

                    # store pervious iterate to compute difference later on
                    S.levels[0].uold[:] = S.levels[0].u[:]

                    if len(S.levels) > 1:  # MLSDC or PFASST
                        S.status.stage = 'IT_UP'
                    else:  # SDC or MSSDC
                        if len(MS) == 1 or self.params.mssdc_jac:  # SDC or parallel MSSDC (Jacobi-like)
                            S.status.stage = 'IT_FINE'
                        else:
                            S.status.stage = 'IT_COARSE'  # serial MSSDC (Gauss-like)
                else:
                    S.levels[0].sweep.compute_end_point()
                    self.hooks.post_step(step=S, level_number=0)
                    S.status.stage = 'DONE'

            return MS

        elif stage == 'IT_FINE':
            # do fine sweep for all steps (virtually parallel)

            for S in MS_active:
                S.levels[0].status.sweep = 0

            for k in range(self.nsweeps[0]):

                for S in MS_active:
                    S.levels[0].status.sweep += 1

                for S in MS_active:
                    # send updated values forward
                    self.hooks.pre_comm(step=S, level_number=0)
                    if self.params.fine_comm and not S.status.last:
                        self.logger.debug('Process %2i provides data on level %2i with tag %s'
                                          % (S.status.slot, 0, S.status.iter))
                        self.send(S.levels[0], tag=(0, S.status.iter, S.status.slot))

                    # # receive values
                    if self.params.fine_comm and not S.status.prev_done and not S.status.first:
                        self.logger.debug('Process %2i receives from %2i on level %2i with tag %s' %
                                          (S.status.slot, S.prev.status.slot, 0, S.status.iter))
                        self.recv(S.levels[0], S.prev.levels[0], tag=(0, S.status.iter, S.prev.status.slot))
                    self.hooks.post_comm(step=S, level_number=0, add_to_stats=(k == self.nsweeps[0] - 1))

                for S in MS_active:
                    # standard sweep workflow: update nodes, compute residual, log progress
                    self.hooks.pre_sweep(step=S, level_number=0)
                    S.levels[0].sweep.update_nodes()
                    S.levels[0].sweep.compute_residual()
                    self.hooks.post_sweep(step=S, level_number=0)

            for S in MS_active:
                # update stage
                S.status.stage = 'IT_CHECK'

            return MS

        elif stage == 'IT_UP':
            # go up the hierarchy from finest to coarsest level (parallel)

            for S in MS_active:

                S.transfer(source=S.levels[0], target=S.levels[1])

            for l in range(1, self.nlevels - 1):

                # sweep on middle levels (not on finest, not on coarsest, though)

                for k in range(self.nsweeps[l]):

                    for S in MS_active:

                        # send updated values forward
                        self.hooks.pre_comm(step=S, level_number=l)
                        if self.params.fine_comm and not S.status.last:
                            self.logger.debug('Process %2i provides data on level %2i with tag %s'
                                              % (S.status.slot, l, S.status.iter))
                            self.send(S.levels[l], tag=(l, S.status.iter, S.status.slot))

                        # # receive values
                        if self.params.fine_comm and not S.status.prev_done and not S.status.first:
                            self.logger.debug('Process %2i receives from %2i on level %2i with tag %s' %
                                              (S.status.slot, S.prev.status.slot, l, S.status.iter))
                            self.recv(S.levels[l], S.prev.levels[l], tag=(l, S.status.iter, S.prev.status.slot))
                        self.hooks.post_comm(step=S, level_number=l)

                    for S in MS_active:

                        self.hooks.pre_sweep(step=S, level_number=l)
                        S.levels[l].sweep.update_nodes()
                        S.levels[l].sweep.compute_residual()
                        self.hooks.post_sweep(step=S, level_number=l)

                for S in MS_active:
                    # transfer further up the hierarchy
                    S.transfer(source=S.levels[l], target=S.levels[l + 1])

            for S in MS_active:
                # update stage
                S.status.stage = 'IT_COARSE'

            return MS

        elif stage == 'IT_COARSE':
            # sweeps on coarsest level (serial/blocking)

            for S in MS_active:

                # receive from previous step (if not first)
                self.hooks.pre_comm(step=S, level_number=len(S.levels) - 1)
                if not S.status.first and not S.status.prev_done:
                    self.logger.debug('Process %2i receives from %2i on level %2i with tag %s' %
                                      (S.status.slot, S.prev.status.slot, len(S.levels) - 1, S.status.iter))
                    self.recv(S.levels[-1], S.prev.levels[-1], tag=(len(S.levels), S.status.iter, S.prev.status.slot))
                self.hooks.post_comm(step=S, level_number=len(S.levels) - 1)

                # do the sweep
                self.hooks.pre_sweep(step=S, level_number=len(S.levels) - 1)
                S.levels[-1].sweep.update_nodes()
                S.levels[-1].sweep.compute_residual()
                self.hooks.post_sweep(step=S, level_number=len(S.levels) - 1)

                # send to succ step
                self.hooks.pre_comm(step=S, level_number=len(S.levels) - 1)
                if not S.status.last:
                    self.logger.debug('Process %2i provides data on level %2i with tag %s'
                                      % (S.status.slot, len(S.levels) - 1, S.status.iter))
                    self.send(S.levels[-1], tag=(len(S.levels), S.status.iter, S.status.slot))
                self.hooks.post_comm(step=S, level_number=len(S.levels) - 1, add_to_stats=True)

                # update stage
                if len(S.levels) > 1:  # MLSDC or PFASST
                    S.status.stage = 'IT_DOWN'
                else:  # MSSDC
                    S.status.stage = 'IT_CHECK'

            return MS

        elif stage == 'IT_DOWN':
            # prolong corrections down to finest level (parallel)

            for l in range(self.nlevels - 1, 0, -1):

                for S in MS_active:
                    # prolong values
                    S.transfer(source=S.levels[l], target=S.levels[l - 1])

                # on middle levels: do communication and sweep as usual
                if l - 1 > 0:

                    for k in range(self.nsweeps[l - 1]):

                        for S in MS_active:

                            # send updated values forward
                            self.hooks.pre_comm(step=S, level_number=l - 1)
                            if self.params.fine_comm and not S.status.last:
                                self.logger.debug('Process %2i provides data on level %2i with tag %s'
                                                  % (S.status.slot, l - 1, S.status.iter))
                                self.send(S.levels[l - 1], tag=(l - 1, S.status.iter, S.status.slot))

                            # # receive values
                            if self.params.fine_comm and not S.status.prev_done and not S.status.first:
                                self.logger.debug('Process %2i receives from %2i on level %2i with tag %s' %
                                                  (S.status.slot, S.prev.status.slot, l - 1, S.status.iter))
                                self.recv(S.levels[l - 1], S.prev.levels[l - 1], tag=(l - 1, S.status.iter,
                                                                                      S.prev.status.slot))
                            self.hooks.post_comm(step=S, level_number=l - 1,
                                                 add_to_stats=(k == self.nsweeps[l - 1] - 1))

                        for S in MS_active:

                            self.hooks.pre_sweep(step=S, level_number=l - 1)
                            S.levels[l - 1].sweep.update_nodes()
                            S.levels[l - 1].sweep.compute_residual()
                            self.hooks.post_sweep(step=S, level_number=l - 1)

            for S in MS_active:
                # update stage
                S.status.stage = 'IT_FINE'

            return MS

        else:

            raise ControllerError('Unknown stage, got %s' % stage)
