import itertools
import copy as cp
import numpy as np
import dill

from pySDC.core.Controller import controller
from pySDC.core import Step as stepclass
from pySDC.core.Errors import ControllerError, CommunicationError, ParameterError
from pySDC.implementations.controller_classes.error_estimator import get_ErrorEstimator_nonMPI


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
            self.logger.warning('you have specified a predictor type but only a single level.. '
                                'predictor will be ignored')

        # prepare variables to do with error estimation and resilience
        self.params.use_embedded_estimate = self.params.use_embedded_estimate or self.params.use_adaptivity or\
            self.params.use_HotRod
        self.params.use_extrapolation_estimate = self.params.use_extrapolation_estimate or self.params.use_HotRod
        self.store_uold = self.params.use_iteration_estimator or self.params.use_embedded_estimate
        if self.params.use_adaptivity:
            if 'e_tol' not in description['level_params'].keys():
                raise ParameterError('Please supply "e_tol" in the level parameters')
            if 'restol' in description['level_params'].keys():
                if description['level_params']['restol'] > 0:
                    description['level_params']['restol'] = 0
                    self.logger.warning(f'I want to do always maxiter={description["step_params"]["maxiter"]} iteration\
s to have a constant order in time for adaptivity. Setting restol=0')
        if self.params.use_HotRod and self.params.HotRod_tol == np.inf:
            self.logger.warning('Hot Rod needs a detection threshold, which is now set to infinity, such that a restart\
 is never triggered!')
        self.error_estimator = get_ErrorEstimator_nonMPI(self)

    def check_iteration_estimator(self, MS):
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
                Kest_loc = np.log(S.params.errtol / alpha) / np.log(Ltilde_loc) * 1.05  # Safety factor!
                self.logger.debug(f'LOCAL: {L.time:8.4f}, {S.status.iter}: {int(np.ceil(Kest_loc))}, '
                                  f'{Ltilde_loc:8.6e}, {Kest_loc:8.6e}, {Ltilde_loc ** S.status.iter * alpha:8.6e}')
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
            if True in restarts:  # restart part of the block
                # find the place after which we need to restart
                restart_at = np.where(restarts)[0][0]

                # store values in the current block that don't need restarting
                if restart_at > 0:
                    self.error_estimator.store_values(MS_active[:restart_at])

                # initial condition to next block is initial condition of step that needs restarting
                uend = self.MS[restart_at].levels[0].u[0]
                time[active_slots[0]] = time[restart_at]

            else:  # move on to next block
                self.error_estimator.store_values(MS_active)

                # initial condition for next block is last solution of current block
                uend = self.MS[active_slots[-1]].levels[0].uend
                time[active_slots[0]] = time[active_slots[-1]] + self.MS[active_slots[-1]].dt

            self.update_step_sizes(active_slots)

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
            'IT_UP': self.it_up
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

            if self.store_uold:
                # store pervious iterate to compute difference later on
                S.levels[0].uold[:] = S.levels[0].u[:]

            # update stage
            if len(S.levels) > 1:  # MLSDC or PFASST with predict
                S.status.stage = 'PREDICT'
            else:
                S.status.stage = 'IT_CHECK'

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

                    self.hooks.pre_comm(step=S, level_number=len(S.levels) - 1)
                    # send updated values on coarsest level
                    self.logger.debug('Process %2i provides data on level %2i with tag %s -- PREDICT'
                                      % (S.status.slot, len(S.levels) - 1, 0))
                    self.send(S.levels[-1], tag=(len(S.levels), 0, S.status.slot))
                    self.hooks.post_comm(step=S, level_number=len(S.levels) - 1)

                # loop over last steps: [2,3,4], [3,4], [4]
                for p in range(q + 1, len(local_MS_running)):
                    S = local_MS_running[p]
                    # receive values sent during previous sweep
                    self.hooks.pre_comm(step=S, level_number=len(S.levels) - 1)
                    self.logger.debug('Process %2i receives from %2i on level %2i with tag %s -- PREDICT' %
                                      (S.status.slot, S.prev.status.slot, len(S.levels) - 1, 0))
                    self.recv(S.levels[-1], S.prev.levels[-1], tag=(len(S.levels), 0, S.prev.status.slot))
                    self.hooks.post_comm(step=S, level_number=len(S.levels) - 1,
                                         add_to_stats=(p == len(local_MS_running) - 1))

            # loop over all steps
            for S in local_MS_running:

                # interpolate back to finest level
                for l in range(len(S.levels) - 1, 0, -1):
                    S.transfer(source=S.levels[l], target=S.levels[l - 1])

                # send updated values forward
                self.hooks.pre_comm(step=S, level_number=0)
                if not S.status.last:
                    self.logger.debug('Process %2i provides data on level %2i with tag %s'
                                      % (S.status.slot, 0, S.status.iter))
                    self.send(S.levels[0], tag=(0, S.status.iter, S.status.slot))

                # receive values
                if not S.status.prev_done and not S.status.first:
                    self.logger.debug('Process %2i receives from %2i on level %2i with tag %s' %
                                      (S.status.slot, S.prev.status.slot, 0, S.status.iter))
                    self.recv(S.levels[0], S.prev.levels[0], tag=(0, S.status.iter, S.prev.status.slot))
                self.hooks.post_comm(step=S, level_number=0)

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
            self.hooks.pre_comm(step=S, level_number=0)
            if not S.status.last:
                self.logger.debug('Process %2i provides data on level %2i with tag %s'
                                  % (S.status.slot, 0, S.status.iter))
                self.send(S.levels[0], tag=(0, S.status.iter, S.status.slot))

            # receive values
            if not S.status.prev_done and not S.status.first:
                self.logger.debug('Process %2i receives from %2i on level %2i with tag %s' %
                                  (S.status.slot, S.prev.status.slot, 0, S.status.iter))
                self.recv(S.levels[0], S.prev.levels[0], tag=(0, S.status.iter, S.prev.status.slot))
            self.hooks.post_comm(step=S, level_number=0)

            S.levels[0].sweep.compute_residual()

        if self.params.use_iteration_estimator:
            self.check_iteration_estimator(local_MS_running)

        self.error_estimator.estimate(local_MS_running)

        if self.params.use_adaptivity:
            self.adaptivity(local_MS_running)

        self.resilience(local_MS_running)

        for S in local_MS_running:

            S.status.done = self.check_convergence(S)

            if S.status.iter > 0:
                self.hooks.post_iteration(step=S, level_number=0)

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

                if self.store_uold:
                    # store previous iterate to compute difference later on
                    S.levels[0].uold[:] = S.levels[0].u[:]

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
                self.hooks.pre_comm(step=S, level_number=0)
                if not S.status.last:
                    self.logger.debug('Process %2i provides data on level %2i with tag %s'
                                      % (S.status.slot, 0, S.status.iter))
                    self.send(S.levels[0], tag=(0, S.status.iter, S.status.slot))

                # # receive values
                if not S.status.prev_done and not S.status.first:
                    self.logger.debug('Process %2i receives from %2i on level %2i with tag %s' %
                                      (S.status.slot, S.prev.status.slot, 0, S.status.iter))
                    self.recv(S.levels[0], S.prev.levels[0], tag=(0, S.status.iter, S.prev.status.slot))
                self.hooks.post_comm(step=S, level_number=0, add_to_stats=(k == self.nsweeps[0] - 1))

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
                    self.hooks.pre_comm(step=S, level_number=l)
                    if not S.status.last:
                        self.logger.debug('Process %2i provides data on level %2i with tag %s'
                                          % (S.status.slot, l, S.status.iter))
                        self.send(S.levels[l], tag=(l, S.status.iter, S.status.slot))

                    # # receive values
                    if not S.status.prev_done and not S.status.first:
                        self.logger.debug('Process %2i receives from %2i on level %2i with tag %s' %
                                          (S.status.slot, S.prev.status.slot, l, S.status.iter))
                        self.recv(S.levels[l], S.prev.levels[l], tag=(l, S.status.iter, S.prev.status.slot))
                    self.hooks.post_comm(step=S, level_number=l)

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
                        self.hooks.pre_comm(step=S, level_number=l - 1)
                        if not S.status.last:
                            self.logger.debug('Process %2i provides data on level %2i with tag %s'
                                              % (S.status.slot, l - 1, S.status.iter))
                            self.send(S.levels[l - 1], tag=(l - 1, S.status.iter, S.status.slot))

                        # # receive values
                        if not S.status.prev_done and not S.status.first:
                            self.logger.debug('Process %2i receives from %2i on level %2i with tag %s' %
                                              (S.status.slot, S.prev.status.slot, l - 1, S.status.iter))
                            self.recv(S.levels[l - 1], S.prev.levels[l - 1], tag=(l - 1, S.status.iter,
                                                                                  S.prev.status.slot))
                        self.hooks.post_comm(step=S, level_number=l - 1,
                                             add_to_stats=(k == self.nsweeps[l - 1] - 1))

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

    def resilience(self, local_MS_running):
        """
        Call various functions that are supposed to provide some sort of resilience from here
        """

        if self.params.use_HotRod:
            self.hotrod(local_MS_running)

        # a step gets restarted because it wants to or because any earlier step wants to
        restart = False
        for p in range(len(local_MS_running)):
            restart = restart or local_MS_running[p].status.restart
            local_MS_running[p].status.restart = restart

    def hotrod(self, local_MS_running):
        """
        See for the reference:
        Lightweight and Accurate Silent Data Corruption Detection in Ordinary Differential Equation Solvers,
        Guhur et al. 2016, Springer. DOI: 10.1007/978-3-319-43659-3_47
        """
        for i in range(len(local_MS_running)):
            S = local_MS_running[i]
            if S.status.iter == S.params.maxiter:
                for l in S.levels:
                    # throw away the final sweep to match the error estimates
                    l.u[:] = l.uold[:]

                    # check if a fault is detected
                    if None not in [l.status.error_extrapolation_estimate, l.status.error_embedded_estimate]:
                        diff = l.status.error_extrapolation_estimate - l.status.error_embedded_estimate
                        if diff > self.params.HotRod_tol:
                            S.status.restart = True

    def adaptivity(self, MS):
        """
        Method to compute time step size adaptively based on embedded error estimate.
        Adaptivity requires you to know the order of the scheme, which you can also know for Jacobi, but it works
        differently.
        """
        if len(MS) > 1 and self.params.mssdc_jac:
            raise NotImplementedError('Adaptivity for multi step SDC only implemented for block Gauss-Seidel')

        # loop through steps and compute local error and optimal step size from there
        for i in range(len(MS)):
            S = MS[i]

            # check if we performed the desired amount of sweeps
            if S.status.iter < S.params.maxiter:
                continue

            L = S.levels[0]

            # compute next step size
            order = S.status.iter  # embedded error estimate is same order as time marching
            assert L.status.error_embedded_estimate is not None, 'Make sure to estimate the embedded error before call\
ing adaptivity!'

            L.status.dt_new = L.params.dt * 0.9 * (L.params.e_tol / L.status.error_embedded_estimate)**(1. / order)

            # check whether to move on or restart
            if L.status.error_embedded_estimate >= L.params.e_tol:
                S.status.restart = True

    def update_step_sizes(self, active_slots):
        """
        Update the step sizes computed in adaptivity or wherever here, since this can get arbitrarily elaborate
        """
        # figure out where the block is restarted
        restarts = [self.MS[p].status.restart for p in active_slots]
        if True in restarts:
            restart_at = np.where(restarts)[0][0]
        else:
            restart_at = len(restarts) - 1

        # record the step sizes to restart with
        new_steps = [None] * len(self.MS[restart_at].levels)
        for i in range(len(self.MS[restart_at].levels)):
            l = self.MS[restart_at].levels[i]
            new_steps[i] = l.status.dt_new if l.status.dt_new is not None else l.params.dt

        # spread the step sizes to all levels
        for j in range(len(active_slots)):
            # get slot number
            p = active_slots[j]

            for i in range(len(self.MS[p].levels)):
                self.MS[p].levels[i].params.dt = new_steps[i]
