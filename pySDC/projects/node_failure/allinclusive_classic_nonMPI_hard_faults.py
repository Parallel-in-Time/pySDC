from pySDC.implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI

from pySDC.projects.node_failure.emulate_hard_faults import hard_fault_injection


class allinclusive_classic_nonMPI_hard_faults(allinclusive_classic_nonMPI):
    """

    PFASST controller, running serialized version of PFASST in classical style, allows node failure before fine sweep

    """

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
                print("Don't know what to do after spread, aborting")
                exit()

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

            S = hard_fault_injection(S)

            # standard sweep workflow: update nodes, compute residual, log progress
            self.hooks.pre_sweep(step=S, level_number=0)
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
                S.levels[l].sweep.update_nodes()
                S.levels[l].sweep.compute_residual()
                self.hooks.post_sweep(step=S, level_number=l)

                # send if last send succeeded on this level (otherwise: abort with error (FIXME))
                if not S.levels[l].tag or S.status.last or S.next.status.done:
                    if self.params.fine_comm:
                        self.send(S.levels[l], tag=True)
                else:
                    print('SEND ERROR', l, S.levels[l].tag)
                    exit()

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
                    self.recv(S.levels[-1], S.prev.levels[-1])
                    S.prev.levels[-1].tag = False
                    if len(S.levels) > 1 or num_procs > 1:
                        S.status.stage = 'IT_COARSE_SWEEP'
                    else:
                        print('you should not be here')
                        exit()
                else:
                    S.status.stage = 'IT_COARSE_RECV'
            else:
                if len(S.levels) > 1 or num_procs > 1:
                    S.status.stage = 'IT_COARSE_SWEEP'
                else:
                    print('you should not be here')
                    exit()
            # return
            return S

        elif stage == 'IT_COARSE_SWEEP':
            # coarsest sweep

            # standard sweep workflow: update nodes, compute residual, log progress
            self.hooks.pre_sweep(step=S, level_number=len(S.levels) - 1)
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

                # if applicable, try to receive values from IT_UP, otherwise abort (fixme)
                if self.params.fine_comm and not S.status.first and not S.prev.status.done:
                    if S.prev.levels[l - 1].tag:
                        self.recv(S.levels[l - 1], S.prev.levels[l - 1])
                        S.prev.levels[l - 1].tag = False
                    else:
                        print('RECV ERROR DOWN')
                        exit()

                # prolong values
                S.transfer(source=S.levels[l], target=S.levels[l - 1])

                # on middle levels: do sweep as usual
                if l - 1 > 0:
                    self.hooks.pre_sweep(step=S, level_number=l - 1)
                    S.levels[l - 1].sweep.update_nodes()
                    S.levels[l - 1].sweep.compute_residual()
                    self.hooks.post_sweep(step=S, level_number=l - 1)

            # update stage and return
            S.status.stage = 'IT_FINE_SWEEP'
            return S

        else:

            # fixme: use meaningful error object here
            print('Something is wrong here, you should have hit one case statement!')
            exit()
