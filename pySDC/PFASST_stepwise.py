import itertools
import copy as cp
import numpy as np

from pySDC.Stats import stats

from pySDC.PFASST_helper import *


def run_pfasst(MS,u0,t0,dt,Tend):
    """
    Main driver for running the serial version of SDC, MLSDC and PFASST (virtual parallelism)

    Args:
        MS: block of steps (list)
        u0: initial values on the finest level
        t0: initial time
        dt: step size (could be changed here e.g. for adaptivity)
        Tend: end time

    Returns:
        end values on the finest level
        stats object containing statistics for each step, each level and each iteration
    """

    # fixme: add ring parallelization as before
    # fixme: use error classes for send/recv and stage errors
    # fixme: last need to be able to send even if values have not been fetched yet (ring!)

    # some initializations
    uend = None
    num_procs = len(MS)

    # initial ordering of the steps: 0,1,...,Np-1
    slots = [p for p in range(num_procs)]

    # initialize time variables of each step
    for p in slots:
        MS[p].status.dt = dt # could have different dt per step here
        MS[p].status.time = t0 + sum(MS[j].status.dt for j in range(p))
        MS[p].status.step = p

    # determine which steps are still active (time < Tend)
    active = [MS[p].status.time < Tend - np.finfo(float).eps for p in slots]
    # compress slots according to active steps, i.e. remove all steps which have times above Tend
    active_slots = list(itertools.compress(slots, active))

    # initialize block of steps with u0
    MS = restart_block(MS,active_slots,u0)

    # call pre-start hook
    MS[active_slots[0]].levels[0].hooks.dump_pre(MS[p].status)

    # main loop: as long as at least one step is still active (time < Tend), do something
    while any(active):

        # loop over all active steps (in the correct order)
        for p in active_slots:
            # print(p,MS[p].status.stage)
            MS[p] = pfasst(MS[p])

        # if all active steps are done (for block-parallelization, need flag to distinguish (FIXME))
        if all([MS[p].status.done for p in active_slots]):

            # uend is uend of the last active step in the list
            uend = MS[active_slots[-1]].levels[0].uend # FIXME: only true for non-ring-parallelization?

            # determine new set of active steps and compress slots accordingly
            active = [MS[p].status.time+num_procs*MS[p].status.dt < Tend - np.finfo(float).eps for p in slots]
            active_slots = list(itertools.compress(slots, active))

            # increment timings for now active steps
            for p in active_slots:
                MS[p].status.time += num_procs*MS[p].status.dt
                MS[p].status.step += num_procs
            # restart active steps (reset all values and pass uend to u0)
            MS = restart_block(MS,active_slots,uend)

        # fixme: for ring parallelization
        # update first and last
        # update slots
        # update pred_cnt

        # This is only for ring-parallelization
        # indx = np.argsort([MS[p].time for p in slots])
        # slots = slots[indx]

        # active = [MS[p].time < Tend for p in slots]

        # if all(not active[p] for p in slots):
        #     for p in slots:
        #         MS[p].time =

    return uend,stats.return_stats()


def restart_block(MS,active_slots,u0):
    """
    Helper routine to reset/restart block of (active) steps

    Args:
        MS: block of (all) steps
        active_slots: list of active steps
        u0: initial value to distribute across the steps

    Returns:
        block of (all) steps
    """

    # loop over active slots (not directly, since we need the previous entry as well)
    for j in range(len(active_slots)):

            # get slot number
            p = active_slots[j]

            # store current slot number for diagnostics
            MS[p].status.slot = p
            # store link to previous step
            MS[p].prev = MS[active_slots[j-1]]
            # resets step
            MS[p].reset_step()
            # determine whether I am the first and/or last in line
            MS[p].status.first = active_slots.index(p) == 0
            MS[p].status.last = active_slots.index(p) == len(active_slots)-1
            # intialize step with u0
            MS[p].init_step(u0)
            # reset some values
            MS[p].status.done = False
            MS[p].status.pred_cnt = active_slots.index(p)+1 # fixme: does this also work for ring-parallelization?
            MS[p].status.iter = 0
            MS[p].status.stage = 'SPREAD'
            for l in MS[p].levels:
                l.tag = False

    return MS


def recv(target,source):
    """
    Receive function

    Args:
        target: level which will receive the values
        source: level which initiated the send
    """
    # simply do a deepcopy of the values uend to become the new u0 at the target
    target.u[0] = target.prob.dtype_u(source.uend)


def send(source,tag):
    """
    Send function

    Args:
        source: level which has the new values
        tag: signal new values
    """
    # sending here means computing uend ("one-sided communication")
    source.sweep.compute_end_point()
    source.tag = cp.deepcopy(tag)


def pfasst(S):
    """
    Main function including the stages of SDC, MLSDC and PFASST (the "controller")

    For the workflow of this controller, check out the picture

    Args:
        S: current step

    Returns:
        current step
    """

    # if S is done, stop right here
    if S.status.done:
        return S

    # otherwise: read out stage of S and act accordingly
    for case in switch(S.status.stage):


        if case('SPREAD'):
            # first stage: spread values
            S.levels[0].hooks.pre_step(S.status)

            # call predictor from sweeper
            S.levels[0].sweep.predict()

            # update stage and return
            if len(S.levels) > 1 and S.params.predict:
                S.status.stage = 'PREDICT_RESTRICT'
            else:
                S.status.stage = 'IT_FINE_SWEEP'
            return S


        if case('PREDICT_RESTRICT'):
            # go to coarsest level via transfer

            for l in range(1,len(S.levels)):
                S.transfer(source=S.levels[l-1],target=S.levels[l])

            # update stage and return
            S.status.stage = 'PREDICT_SWEEP'
            return S


        if case('PREDICT_SWEEP'):
            # do a (serial) sweep on coarsest level

            # receive new values from previous step (if not first step)
            if not S.status.first:
                if S.prev.levels[-1].tag:
                    recv(S.levels[-1],S.prev.levels[-1])
                    # reset tag to signal successful receive
                    S.prev.levels[-1].tag = False

            # do the sweep with (possibly) new values
            S.levels[-1].sweep.update_nodes()

            # update stage and return
            S.status.stage = 'PREDICT_SEND'
            return S


        if case('PREDICT_SEND'):
            # send updated values on coarsest level

            # send new values forward, if previous send was successful (otherwise: try again)
            if not S.status.last:
                if not S.levels[-1].tag:
                    send(S.levels[-1],tag=True)
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


        if case('PREDICT_INTERP'):
            # prolong back to finest level

            for l in range(len(S.levels)-1,0,-1):
                S.transfer(source=S.levels[l],target=S.levels[l-1])

            # uodate stage and return
            S.status.stage = 'IT_FINE_SWEEP'
            return S


        if case('IT_FINE_SWEEP'):
            # do sweep on finest level

            # increment iteration count here (and only here)
            S.status.iter += 1

            # standard sweep workflow: update nodes, compute residual, log progress
            S.levels[0].sweep.update_nodes()
            S.levels[0].sweep.compute_residual()
            S.levels[0].hooks.dump_sweep(S.status)

            S.levels[0].hooks.dump_iteration(S.status)

            # update stage and return
            S.status.stage = 'IT_FINE_SEND'

            return S


        if case('IT_FINE_SEND'):
            # send forward values on finest level

            # if last send succeeded on this level or if last rank, send new values (otherwise: try again)
            if not S.levels[0].tag or S.status.last:
                if S.params.fine_comm:
                    send(S.levels[0],tag=True)
                S.status.stage = 'IT_CHECK'
            else:
                S.status.stage = 'IT_FINE_SEND'
            # return
            return S


        if case('IT_CHECK'):
            # check whether to stop iterating

            S.status.done = check_convergence(S)

            # if the previous step is still iterating but I am done, un-do me to still forward values
            if not S.status.first and S.status.done and not S.prev.status.done:
                S.status.done = False

            # if I am done, signal accordingly, otherwise proceed
            if S.status.done:
                S.levels[0].sweep.compute_end_point()
                S.levels[0].hooks.dump_step(S.status)
                S.status.stage = 'DONE'
            else:
                if len(S.levels) > 1:
                    S.status.stage = 'IT_UP'
                else:
                    S.status.stage = 'IT_COARSE_RECV'
            # return
            return S


        if case('IT_UP'):
            # go up the hierarchy from finest to coarsest level

            S.transfer(source=S.levels[0],target=S.levels[1])

            # sweep and send on middle levels (not on finest, not on coarsest, though)
            for l in range(1,len(S.levels)-1):
                S.levels[l].sweep.update_nodes()
                S.levels[l].sweep.compute_residual()
                S.levels[l].hooks.dump_sweep(S.status)

                # send if last send succeeded on this level (otherwise: abort with error (FIXME))
                if not S.levels[l].tag or S.status.last:
                    if S.params.fine_comm:
                        send(S.levels[l],tag=True)
                else:
                    print('SEND ERROR',l,p,S.levels[l].tag)
                    exit()

                # transfer further up the hierarchy
                S.transfer(source=S.levels[l],target=S.levels[l+1])

            # update stage and return
            S.status.stage = 'IT_COARSE_RECV'
            return S


        if case('IT_COARSE_RECV'):
            # receive on coarsest level

            # rather complex logic here...
            # if I am not the first in line and if the first is not done yet, try to receive
            # otherwise: proceed, no receiving possible/necessary
            if not S.status.first and not S.prev.status.done:
                # try to receive and the progress (otherwise: try again)
                if S.prev.levels[-1].tag:
                    recv(S.levels[-1],S.prev.levels[-1])
                    S.prev.levels[-1].tag = False
                    if len(S.levels) > 1:
                        S.status.stage = 'IT_COARSE_SWEEP'
                    else:
                        S.status.stage = 'IT_FINE_SWEEP'
                else:
                    S.status.stage = 'IT_COARSE_RECV'
            else:
                if len(S.levels) > 1:
                    S.status.stage = 'IT_COARSE_SWEEP'
                else:
                    S.status.stage = 'IT_FINE_SWEEP'
            # return
            return S


        if case('IT_COARSE_SWEEP'):
            # coarsest sweep

            # standard sweep workflow: update nodes, compute residual, log progress
            S.levels[-1].sweep.update_nodes()
            S.levels[-1].sweep.compute_residual()

            S.levels[-1].hooks.dump_sweep(S.status)

            # update stage and return
            S.status.stage = 'IT_COARSE_SEND'
            return S


        if case('IT_COARSE_SEND'):
            # send forward coarsest values

            # try to send new values (if old ones have not been picked up yet, retry)
            if not S.levels[-1].tag or S.status.last:
                send(S.levels[-1],tag=True)
                S.status.stage = 'IT_DOWN'
            else:
                S.status.stage = 'IT_COARSE_SEND'
            # return
            return S


        if case('IT_DOWN'):
            # prolong corrections own to finest level

            # receive and sweep on middle levels (except for coarsest level)
            for l in range(len(S.levels)-1,0,-1):

                # if applicable, try to receive values from IT_UP, otherwise abort (fixme)
                if S.params.fine_comm and not S.status.first and not S.prev.status.done:
                    if S.prev.levels[l-1].tag:
                        recv(S.levels[l-1],S.prev.levels[l-1])
                        S.prev.levels[l-1].tag = False
                    else:
                        print('RECV ERROR DOWN')
                        exit()

                # prolong values
                S.transfer(source=S.levels[l],target=S.levels[l-1])

                # on middle levels: do sweep as usual
                if l-1 > 0:
                    S.levels[l-1].sweep.update_nodes()
                    S.levels[l-1].sweep.compute_residual()
                    S.levels[l-1].hooks.dump_sweep(S.status)

            # update stage and return
            S.status.stage = 'IT_FINE_SWEEP'
            return S

        #fixme: use meaningful error object here
        print('Something is wrong here, you should have hit one case statement!')
        exit()
    #fixme: use meaningful error object here
    print('Something is wrong here, you should have hit one case statement!')
    exit()




