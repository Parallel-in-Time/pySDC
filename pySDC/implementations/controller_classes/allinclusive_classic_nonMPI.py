import itertools
import copy as cp
import numpy as np

from pySDC.core.Controller import controller
from pySDC.core import Step as stepclass
from pySDC.core.Errors import CommunicationError, ControllerError


class allinclusive_classic_nonMPI(controller):
    """
    PFASST controller, running serialized version of PFASST in classical style
    """

    def __init__(self, num_procs, controller_params, description):
        """
        Initialization routine for PFASST controller

        Args:
            num_procs: number of parallel time steps (still serial, though), can be 1
            controller_params: parameter set for the controller and the step class
            description: all the parameters to set up the rest (levels, problems, transfer, ...)
        """

        # call parent's initialization routine
        super(allinclusive_classic_nonMPI, self).__init__(controller_params)

        self.MS = []
        # simply append step after step and generate the hierarchies
        for p in range(num_procs):
            self.MS.append(stepclass.step(description))

        if self.params.dump_setup:
            self.dump_setup(step=self.MS[0], controller_params=controller_params, description=description)

        num_levels = len(self.MS[0].levels)

        if num_procs > 1 and num_levels > 1:
            for S in self.MS:
                for L in S.levels:
                    if not L.sweep.coll.right_is_node or L.sweep.params.do_coll_update:
                        raise ControllerError("For PFASST to work, we assume uend^k = u_M^k in this controller")

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

        # compress slots according to active steps, i.e. remove all steps which have times above Tend
        active_slots = list(itertools.compress(slots, active))

        # initialize block of steps with u0
        self.restart_block(active_slots, time, u0)

        # call pre-run hook
        for S in self.MS:
            self.hooks.pre_run(step=S, level_number=0)

        # main loop: as long as at least one step is still active (time < Tend), do something
        while any(active):

            # loop over all active steps (in the correct order)
            while not all([self.MS[p].status.done for p in active_slots]):

                for p in active_slots:
                    self.MS[p] = self.pfasst(self.MS[p], len(active_slots))

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
            time: new times for the steps
            u0: initial value to distribute across the steps

        """

        # loop over active slots (not directly, since we need the previous entry as well)
        for j in range(len(active_slots)):

            # get slot number
            p = active_slots[j]

            # store current slot number for diagnostics
            self.MS[p].status.slot = p

            # resets step
            self.MS[p].reset_step()
            # determine whether I am the first and/or last in line
            self.MS[p].status.first = active_slots.index(p) == 0
            if not self.MS[p].status.first:
                # store link to previous step
                self.MS[p].prev = self.MS[active_slots[j - 1]]

            self.MS[p].status.last = active_slots.index(p) == len(active_slots) - 1
            if not self.MS[p].status.last:
                # store link to next step
                self.MS[p].next = self.MS[active_slots[j + 1]]
            # intialize step with u0
            self.MS[p].init_step(u0)
            # reset some values
            self.MS[p].status.done = False
            self.MS[p].status.pred_cnt = active_slots.index(p) + 1
            self.MS[p].status.iter = 1
            self.MS[p].status.stage = 'SPREAD'
            for l in self.MS[p].levels:
                l.tag = False

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
        # uend becomes the new u0 at the target
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

    def pfasst(self, S, num_procs):
        """
        Main function including the stages of SDC, MLSDC and PFASST (the "controller")

        For the workflow of this controller, check out one of our PFASST talks

        Args:
            S: currently active step
            num_procs: number of active processors

        Returns:
            updated step
        """

        # if S is done, stop right here
        if S.status.done:
            return S

        stage = S.status.stage

        self.logger.debug("Process %2i at stage %s" % (S.status.slot, stage))

        if stage == 'SPREAD':
            # first stage: spread values
            self.hooks.pre_step(step=S, level_number=0)

            # call predictor from sweeper
            S.levels[0].sweep.predict()

            # update stage
            if len(S.levels) > 1 and self.params.predict:  # MLSDC or PFASST with predict
                S.status.stage = 'PREDICT_RESTRICT'
            elif len(S.levels) > 1:  # MLSDC or PFASST without predict
                self.hooks.pre_iteration(step=S, level_number=0)
                S.status.stage = 'IT_FINE_SWEEP'
            elif num_procs > 1:  # MSSDC
                self.hooks.pre_iteration(step=S, level_number=0)
                S.status.stage = 'IT_COARSE_SWEEP'
            elif num_procs == 1:  # SDC
                self.hooks.pre_iteration(step=S, level_number=0)
                S.status.stage = 'IT_FINE_SWEEP'
            else:
                raise ControllerError("Don't know what to do after spread, aborting")

            return S

        elif stage == 'PREDICT_RESTRICT':
            # call predictor (serial)

            # go to coarsest level via transfer

            for l in range(1, len(S.levels)):
                S.transfer(source=S.levels[l - 1], target=S.levels[l])

            # update stage and return
            S.status.stage = 'PREDICT_SWEEP'
            return S

        elif stage == 'PREDICT_SWEEP':

            # do a (serial) sweep on coarsest level

            # receive new values from previous step (if not first step)
            if not S.status.first:
                if S.prev.levels[-1].tag:
                    self.logger.debug('Process %2i receives from %2i on level %2i with tag %s -- PREDICT' %
                                      (S.status.slot, S.prev.status.slot, len(S.levels) - 1, True))
                    self.recv(S.levels[-1], S.prev.levels[-1])
                    # reset tag to signal successful receive
                    S.prev.levels[-1].tag = False

            # do the sweep with (possibly) new values
            S.levels[-1].sweep.update_nodes()

            # update stage and return
            S.status.stage = 'PREDICT_SEND'
            return S

        elif stage == 'PREDICT_SEND':
            # send updated values on coarsest level

            # send new values forward, if previous send was successful (otherwise: try again)
            if not S.status.last:
                if not S.levels[-1].tag:
                    self.logger.debug('Process %2i provides data on level %2i with tag %s -- PREDICT'
                                      % (S.status.slot, len(S.levels) - 1, True))
                    self.send(S.levels[-1], tag=True)
                else:
                    S.status.stage = 'PREDICT_SEND'
                    return S

            # decrement counter to determine how many coarse sweeps are necessary
            S.status.pred_cnt -= 1

            # update stage and return
            if S.status.pred_cnt == 0:
                S.status.stage = 'PREDICT_INTERP'
            else:
                S.status.stage = 'PREDICT_SWEEP'
            return S

        elif stage == 'PREDICT_INTERP':
            # prolong back to finest level

            for l in range(len(S.levels) - 1, 0, -1):
                S.transfer(source=S.levels[l], target=S.levels[l - 1])

            # update stage and return
            self.hooks.pre_iteration(step=S, level_number=0)
            S.status.stage = 'IT_FINE_SWEEP'
            return S

        elif stage == 'IT_FINE_SWEEP':
            # do sweep on finest level

            # standard sweep workflow: update nodes, compute residual, log progress
            self.hooks.pre_sweep(step=S, level_number=0)

            for k in range(S.levels[0].params.nsweeps):
                S.levels[0].sweep.update_nodes()
            S.levels[0].sweep.compute_residual()
            self.hooks.post_sweep(step=S, level_number=0)

            # update stage and return
            S.status.stage = 'IT_FINE_SEND'

            return S

        elif stage == 'IT_FINE_SEND':
            # send forward values on finest level

            # if last send succeeded on this level or if last rank, send new values (otherwise: try again)
            if not S.levels[0].tag or S.status.last or S.next.status.done:
                if self.params.fine_comm:
                    self.logger.debug('Process %2i provides data on level %2i with tag %s'
                                      % (S.status.slot, 0, True))
                    self.send(S.levels[0], tag=True)
                S.status.stage = 'IT_CHECK'
            else:
                S.status.stage = 'IT_FINE_SEND'
            # return
            return S

        elif stage == 'IT_CHECK':

            # check whether to stop iterating

            self.hooks.post_iteration(step=S, level_number=0)

            S.status.done = self.check_convergence(S)

            # if the previous step is still iterating but I am done, un-do me to still forward values
            if not S.status.first and S.status.done and (S.prev.status.done is not None and not S.prev.status.done):
                S.status.done = False

            # if I am done, signal accordingly, otherwise proceed
            if S.status.done:
                S.levels[0].sweep.compute_end_point()
                self.hooks.post_step(step=S, level_number=0)
                S.status.stage = 'DONE'
            else:
                # increment iteration count here (and only here)
                S.status.iter += 1
                self.hooks.pre_iteration(step=S, level_number=0)
                if len(S.levels) > 1:
                    S.status.stage = 'IT_UP'
                elif num_procs > 1:  # MSSDC
                    S.status.stage = 'IT_COARSE_RECV'
                elif num_procs == 1:  # SDC
                    S.status.stage = 'IT_FINE_SWEEP'
            # return
            return S

        elif stage == 'IT_UP':
            # go up the hierarchy from finest to coarsest level

            S.transfer(source=S.levels[0], target=S.levels[1])

            # sweep and send on middle levels (not on finest, not on coarsest, though)
            for l in range(1, len(S.levels) - 1):
                self.hooks.pre_sweep(step=S, level_number=l)
                for k in range(S.levels[l].params.nsweeps):
                    S.levels[l].sweep.update_nodes()
                S.levels[l].sweep.compute_residual()
                self.hooks.post_sweep(step=S, level_number=l)

                # send if last send succeeded on this level (otherwise: abort with error)
                if not S.levels[l].tag or S.status.last or S.next.status.done:
                    if self.params.fine_comm:
                        self.logger.debug('Process %2i provides data on level %2i with tag %s'
                                          % (S.status.slot, l, True))
                        self.send(S.levels[l], tag=True)
                else:
                    raise CommunicationError('Sending failed on process %2i, level %2i' % (S.status.slot, l))

                # transfer further up the hierarchy
                S.transfer(source=S.levels[l], target=S.levels[l + 1])

            # update stage and return
            S.status.stage = 'IT_COARSE_RECV'
            return S

        elif stage == 'IT_COARSE_RECV':

            # receive on coarsest level

            # rather complex logic here...
            # if I am not the first in line and if the first is not done yet, try to receive
            # otherwise: proceed, no receiving possible/necessary
            if not S.status.first and not S.prev.status.done:
                # try to receive and the progress (otherwise: try again)
                if S.prev.levels[-1].tag:
                    self.logger.debug('Process %2i receives from %2i on level %2i with tag %s' %
                                      (S.status.slot, S.prev.status.slot, len(S.levels) - 1, True))
                    self.recv(S.levels[-1], S.prev.levels[-1])
                    S.prev.levels[-1].tag = False
                    if len(S.levels) > 1 or num_procs > 1:
                        S.status.stage = 'IT_COARSE_SWEEP'
                    else:
                        raise ControllerError('Stage unclear after coarse send')
                else:
                    S.status.stage = 'IT_COARSE_RECV'
            else:
                if len(S.levels) > 1 or num_procs > 1:
                    S.status.stage = 'IT_COARSE_SWEEP'
                else:
                    raise ControllerError('Stage unclear after coarse send')
            # return
            return S

        elif stage == 'IT_COARSE_SWEEP':
            # coarsest sweep

            # standard sweep workflow: update nodes, compute residual, log progress
            self.hooks.pre_sweep(step=S, level_number=len(S.levels) - 1)
            for k in range(S.levels[-1].params.nsweeps):
                S.levels[-1].sweep.update_nodes()
            S.levels[-1].sweep.compute_residual()

            self.hooks.post_sweep(step=S, level_number=len(S.levels) - 1)

            # update stage and return
            S.status.stage = 'IT_COARSE_SEND'
            return S

        elif stage == 'IT_COARSE_SEND':
            # send forward coarsest values

            # try to send new values (if old ones have not been picked up yet, retry)
            if not S.levels[-1].tag or S.status.last or S.next.status.done:
                self.logger.debug('Process %2i provides data on level %2i with tag %s'
                                  % (S.status.slot, len(S.levels) - 1, True))
                self.send(S.levels[-1], tag=True)
                # update stage
                if len(S.levels) > 1:  # MLSDC or PFASST
                    S.status.stage = 'IT_DOWN'
                else:  # MSSDC
                    S.status.stage = 'IT_CHECK'
            else:
                S.status.stage = 'IT_COARSE_SEND'
            # return
            return S

        elif stage == 'IT_DOWN':
            # prolong corrections own to finest level

            # receive and sweep on middle levels (except for coarsest level)
            for l in range(len(S.levels) - 1, 0, -1):

                # if applicable, try to receive values from IT_UP, otherwise abort
                if self.params.fine_comm and not S.status.first and not S.prev.status.done:
                    if S.prev.levels[l - 1].tag:
                        self.logger.debug('Process %2i receives from %2i on level %2i with tag %s' %
                                          (S.status.slot, S.prev.status.slot, l - 1, True))
                        self.recv(S.levels[l - 1], S.prev.levels[l - 1])
                        S.prev.levels[l - 1].tag = False
                    else:
                        raise CommunicationError('Sending failed during IT_DOWN')

                # prolong values
                S.transfer(source=S.levels[l], target=S.levels[l - 1])

                # on middle levels: do sweep as usual
                if l - 1 > 0:
                    self.hooks.pre_sweep(step=S, level_number=l - 1)
                    for k in range(S.levels[l - 1].params.nsweeps):
                        S.levels[l - 1].sweep.update_nodes()
                    S.levels[l - 1].sweep.compute_residual()
                    self.hooks.post_sweep(step=S, level_number=l - 1)

            # update stage and return
            S.status.stage = 'IT_FINE_SWEEP'
            return S

        else:

            raise ControllerError('Unknown stage, got %s' % S.status.stage)
