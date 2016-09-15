import itertools
import copy as cp
import numpy as np

from pySDC.Controller import controller
from pySDC import Step as stepclass
from pySDC.Stats import stats


class allinclusive_blockwise_nonMPI(controller):
    """

    PFASST controller, running serialized version of PFASST in blocks (MG-style)

    """

    def __init__(self, num_procs, step_params, description):
        """
       Initialization routine for PFASST controller

       Args:
           num_procs: number of parallel time steps (still serial, though), can be 1
           step_params: parameter set for the step class
           description: all the parameters to set up the rest (levels, problems, transfer, ...)
       """

        # call parent's initialization routine
        super(allinclusive_blockwise_nonMPI, self).__init__()

        self.MS = []
        # simply append step after step and generate the hierarchies
        for p in range(num_procs):
            self.MS.append(stepclass.step(step_params))
            self.MS[-1].generate_hierarchy(description)


    def run(self, u0, t0, dt, Tend):
        """
        Main driver for running the serial version of SDC, MSSDC, MLSDC and PFASST (virtual parallelism)

        Args:
           u0: initial values
           t0: starting time
           dt: (initial) time step
           Tend: ending time

        Returns:
            end values on the finest level
            stats object containing statistics for each step, each level and each iteration
        """

        # fixme: use error classes for send/recv and stage errors

        # some initializations
        uend = None
        num_procs = len(self.MS)

        # initial ordering of the steps: 0,1,...,Np-1
        slots = [p for p in range(num_procs)]

        # initialize time variables of each step
        for p in slots:
            self.MS[p].status.dt = dt # could have different dt per step here
            self.MS[p].status.time = t0 + sum(self.MS[j].status.dt for j in range(p))
            self.MS[p].status.step = p

        # determine which steps are still active (time < Tend)
        active = [self.MS[p].status.time < Tend - 10*np.finfo(float).eps for p in slots]

        # compress slots according to active steps, i.e. remove all steps which have times above Tend
        active_slots = list(itertools.compress(slots, active))

        # initialize block of steps with u0
        self.restart_block(active_slots,u0)

        # call pre-start hook
        self.MS[active_slots[0]].levels[0].hooks.dump_pre(self.MS[p].status)

        # main loop: as long as at least one step is still active (time < Tend), do something
        while any(active):

            MS_active = []
            for p in active_slots:
                MS_active.append(self.MS[p])

            MS_active = self.pfasst(MS_active)

            for p in range(len(MS_active)):
                self.MS[active_slots[p]] = MS_active[p]


            # if all active steps are done
            if all([self.MS[p].status.done for p in active_slots]):

                # uend is uend of the last active step in the list
                uend = self.MS[active_slots[-1]].levels[0].uend

                # determine new set of active steps and compress slots accordingly
                active = [self.MS[p].status.time+num_procs*self.MS[p].status.dt < Tend - 10*np.finfo(float).eps for p in slots]
                active_slots = list(itertools.compress(slots, active))

                # increment timings for now active steps
                for p in active_slots:
                    self.MS[p].status.time += num_procs*self.MS[p].status.dt
                    self.MS[p].status.step += num_procs
                # restart active steps (reset all values and pass uend to u0)
                self.restart_block(active_slots,uend)

        return uend,stats.return_stats()


    def restart_block(self,active_slots,u0):
        """
        Helper routine to reset/restart block of (active) steps

        Args:
            active_slots: list of active steps
            u0: initial value to distribute across the steps

        """

        # loop over active slots (not directly, since we need the previous entry as well)
        for j in range(len(active_slots)):

                # get slot number
                p = active_slots[j]

                # store current slot number for diagnostics
                self.MS[p].status.slot = p
                # store link to previous step
                self.MS[p].prev = self.MS[active_slots[j-1]]
                # resets step
                self.MS[p].reset_step()
                # determine whether I am the first and/or last in line
                self.MS[p].status.first = active_slots.index(p) == 0
                self.MS[p].status.last = active_slots.index(p) == len(active_slots)-1
                # intialize step with u0
                self.MS[p].init_step(u0)
                # reset some values
                self.MS[p].status.done = False
                self.MS[p].status.iter = 1
                self.MS[p].status.stage = 'SPREAD'
                for l in self.MS[p].levels:
                    l.tag = None

    @staticmethod
    def recv(target,source,tag=None):
        """
        Receive function

        Args:
            target: level which will receive the values
            source: level which initiated the send
            tag: identifier to check if this message is really for me
        """

        if tag is not None and source.tag != tag:
            print('RECV ERROR',tag,source.tag)
            exit()
        # simply do a deepcopy of the values uend to become the new u0 at the target
        target.u[0] = target.prob.dtype_u(source.uend)
        # re-evaluate f on left interval boundary
        target.f[0] = target.prob.eval_f(target.u[0],target.time)

    @staticmethod
    def send(source,tag):
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

        # loop over all steps
        for S in MS:

            # restrict to coarsest level
            for l in range(1,len(S.levels)):
                S.transfer(source=S.levels[l-1],target=S.levels[l])

        # loop over all steps
        for q in range(len(MS)):

            # loop over last steps: [1,2,3,4], [2,3,4], [3,4], [4]
            for p in range(q,len(MS)):

                S = MS[p]

                # do the sweep with new values
                S.levels[-1].sweep.update_nodes()

                # send updated values on coarsest level
                self.send(S.levels[-1],tag=(len(S.levels),0,S.status.slot))

            # loop over last steps: [2,3,4], [3,4], [4]
            for p in range(q+1,len(MS)):

                S = MS[p]
                # receive values sent during previous sweep
                self.recv(S.levels[-1],S.prev.levels[-1],tag=(len(S.levels),0,S.prev.status.slot))

        # loop over all steps
        for S in MS:

            # interpolate back to finest level
            for l in range(len(S.levels)-1,0,-1):
                S.transfer(source=S.levels[l],target=S.levels[l-1])

        return MS

    def pfasst(self,MS):
        """
        Main function including the stages of SDC, MLSDC and PFASST (the "controller")

        For the workflow of this controller, check out one of our PFASST talks

        Args:
            MS: all active steps

        Returns:
            all active steps
        """

        stage = None
        # if all stages are the same, continue, otherwise abort
        if all(S.status.stage for S in MS):
            stage = MS[0].status.stage
        else:
            print('not all stages are equal, aborting..')
            exit()

        if stage == 'SPREAD':
            # (potentially) serial spreading phase
            for S in MS:

                # first stage: spread values
                S.levels[0].hooks.pre_step(S.status)

                # call predictor from sweeper
                S.levels[0].sweep.predict()

                # update stage
                if (len(S.levels) > 1 and len(MS) > 1) and S.params.predict: # MLSDC or PFASST
                    S.status.stage = 'PREDICT'
                elif len(MS) > 1 and len(S.levels) > 1: # PFASST
                    S.levels[0].hooks.dump_pre_iteration(S.status)
                    S.status.stage = 'IT_FINE'
                elif len(MS) > 1 and len(S.levels) == 1: # MSSDC
                    S.levels[0].hooks.dump_pre_iteration(S.status)
                    S.status.stage = 'IT_COARSE'
                elif len(MS) == 1: # SDC
                    S.levels[0].hooks.dump_pre_iteration(S.status)
                    S.status.stage = 'IT_FINE'
                else:
                    print("Don't know what to do after spread, aborting")
                    exit()

            return MS

        elif stage == 'PREDICT':
            # call predictor (serial)

            MS = self.predictor(MS)

            for S in MS:
                # update stage
                S.levels[0].hooks.dump_pre_iteration(S.status)
                S.status.stage = 'IT_FINE'

            return MS

        elif stage == 'IT_FINE':
            # do fine sweep for all steps (virtually parallel)

            for S in MS:

                # standard sweep workflow: update nodes, compute residual, log progress
                S.levels[0].sweep.update_nodes()
                S.levels[0].sweep.compute_residual()
                S.levels[0].hooks.dump_sweep(S.status)

                # update stage
                S.status.stage = 'IT_CHECK'

            return MS

        elif stage == 'IT_CHECK':

            # check whether to stop iterating (parallel)

            for S in MS:
                # increment iteration count here (and only here)
                S.status.iter += 1
                S.levels[0].hooks.dump_iteration(S.status)
                S.status.done = self.check_convergence(S)

            # if not everyone is ready yet, keep doing stuff
            if not all(S.status.done for S in MS):

                for S in MS:
                    S.status.done = False
                    # multi-level or single-level?
                    if len(S.levels) > 1: # MLSDC or PFASST
                        S.status.stage = 'IT_UP'
                    elif len(MS) > 1: # MSSDC
                        S.status.stage = 'IT_COARSE'
                    elif len(MS) == 1: # SDC
                        S.status.stage = 'IT_FINE'

            else:
                # if everyone is ready, end
                for S in MS:
                    S.levels[0].sweep.compute_end_point()
                    S.levels[0].hooks.dump_step(S.status)
                    S.status.stage = 'DONE'

            return MS

        elif stage == 'IT_UP':
            # go up the hierarchy from finest to coarsest level (parallel)

            for S in MS:

                S.transfer(source=S.levels[0], target=S.levels[1])

                # sweep and send on middle levels (not on finest, not on coarsest, though)
                for l in range(1, len(S.levels) - 1):
                    S.levels[l].sweep.update_nodes()
                    S.levels[l].sweep.compute_residual()
                    S.levels[l].hooks.dump_sweep(S.status)

                    # transfer further up the hierarchy
                    S.transfer(source=S.levels[l], target=S.levels[l + 1])

                # update stage
                S.status.stage = 'IT_COARSE'

            return MS

        elif stage == 'IT_COARSE':
            # sweeps on coarsest level (serial/blocking)

            for S in MS:

                # receive from previous step (if not first)
                if not S.status.first:
                    self.recv(S.levels[-1], S.prev.levels[-1], tag=(len(S.levels), S.status.iter, S.prev.status.slot))

                # do the sweep
                S.levels[-1].sweep.update_nodes()
                S.levels[-1].sweep.compute_residual()
                S.levels[-1].hooks.dump_sweep(S.status)

                # send to succ step
                self.send(S.levels[-1], tag=(len(S.levels), S.status.iter, S.status.slot))

                # update stage
                if len(S.levels) > 1: # MLSDC or PFASST
                    S.status.stage = 'IT_DOWN'
                else: # MSSDC
                    S.status.stage = 'IT_CHECK'


            return MS

        elif stage == 'IT_DOWN':
            # prolong corrections down to finest level (parallel)

            for S in MS:

                # receive and sweep on middle levels (except for coarsest level)
                for l in range(len(S.levels) - 1, 0, -1):

                    # prolong values
                    S.transfer(source=S.levels[l], target=S.levels[l - 1])

                    # send updated values forward
                    if S.params.fine_comm:
                        self.send(S.levels[l - 1], tag=(l - 1, S.status.iter, S.status.slot))

                    # # receive values
                    if S.params.fine_comm and not S.status.first:
                        self.recv(S.levels[l - 1], S.prev.levels[l - 1], tag=(l - 1, S.status.iter, S.prev.status.slot))

                    # on middle levels: do sweep as usual
                    if l - 1 > 0:
                        S.levels[l - 1].sweep.update_nodes()
                        S.levels[l - 1].sweep.compute_residual()
                        S.levels[l - 1].hooks.dump_sweep(S.status)

                # update stage
                S.status.stage = 'IT_FINE'

            return MS

        else:

            #fixme: use meaningful error object here
            print('Something is wrong here, you should have hit one case statement!')
            exit()
