import itertools
import copy as cp
import numpy as np

from pySDC.Stats import stats

from pySDC.PFASST_helper import *


#TODO:
#  - restore MSSDC
#  - stop iterating if done to avoid noise
#  - ring parallelization...?


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

    # fixme: use error classes for send/recv and stage errors

    # some initializations
    uend = None
    num_procs = len(MS)

    if num_procs > 1:
        assert len(MS[0].levels) > 1

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

        MS_active = []
        for p in active_slots:
            MS_active.append(MS[p])

        MS_active = pfasst(MS_active)

        for p in range(len(MS_active)):
            MS[active_slots[p]] = MS_active[p]


        # if all active steps are done
        if all([MS[p].status.done for p in active_slots]):

            # uend is uend of the last active step in the list
            uend = MS[active_slots[-1]].levels[0].uend

            # determine new set of active steps and compress slots accordingly
            active = [MS[p].status.time+num_procs*MS[p].status.dt < Tend - np.finfo(float).eps for p in slots]
            active_slots = list(itertools.compress(slots, active))

            # increment timings for now active steps
            for p in active_slots:
                MS[p].status.time += num_procs*MS[p].status.dt
                MS[p].status.step += num_procs
            # restart active steps (reset all values and pass uend to u0)
            MS = restart_block(MS,active_slots,uend)

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
            MS[p].status.iter = 0
            MS[p].status.stage = 'SPREAD'
            for l in MS[p].levels:
                l.tag = None

    return MS


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


def predictor(MS):
    """
    Predictor function, extracted from the stepwise implementation (will be also used by matrix sweppers)

    Args:
        MS: multiple steps

    Returns:
        block of steps with initial values
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
            send(S.levels[-1],tag=(len(S.levels),0,S.status.slot))

        # loop over last steps: [2,3,4], [3,4], [4]
        for p in range(q+1,len(MS)):

            S = MS[p]
            # receive values sent during previous sweep
            recv(S.levels[-1],S.prev.levels[-1],tag=(len(S.levels),0,S.prev.status.slot))

    # loop over all steps
    for S in MS:

        # interpolate back to finest level
        for l in range(len(S.levels)-1,0,-1):
            S.transfer(source=S.levels[l],target=S.levels[l-1])

    return MS


def pfasst(MS):
    """
    Main function including the stages of SDC, MLSDC and PFASST (the "controller")

    For the workflow of this controller, check out one of our PFASST talks

    Args:
        MS: all active steps

    Returns:
        all active steps
    """

    # if all stages are the same, continue, otherwise abort
    if all(S.status.stage for S in MS):
        stage = MS[0].status.stage
    else:
        print('not all stages are equal, aborting..')
        exit()


    for case in switch(stage):

        if case('SPREAD'):
            # (potentially) serial spreading phase

            for S in MS:

                # first stage: spread values
                S.levels[0].hooks.pre_step(S.status)

                # call predictor from sweeper
                S.levels[0].sweep.predict()

                # update stage
                if len(S.levels) > 1:
                    S.status.stage = 'PREDICT'
                else:
                    S.status.stage = 'IT_FINE'

            return MS


        if case('PREDICT'):
            # call predictor (serial)

            MS = predictor(MS)

            for S in MS:
                # update stage
                S.status.stage = 'IT_FINE'

            return MS


        if case('IT_FINE'):
            # do fine sweep for all steps (virtually parallel)

            for S in MS:
                # increment iteration count here (and only here)
                S.status.iter += 1

                # standard sweep workflow: update nodes, compute residual, log progress
                S.levels[0].sweep.update_nodes()
                S.levels[0].sweep.compute_residual()
                S.levels[0].hooks.dump_sweep(S.status)

                S.levels[0].hooks.dump_iteration(S.status)

                # send updated values forward (non-blocking)
                if S.params.fine_comm:
                    send(S.levels[0],tag=(0,S.status.iter,S.status.slot))

                # update stage
                S.status.stage = 'IT_CHECK'

            return MS


        if case('IT_CHECK'):
            # check whether to stop iterating (parallel)

            for S in MS:
                S.status.done = check_convergence(S)

            # if not everyone is ready yet, keep doing stuff
            if not all(S.status.done for S in MS):

                for S in MS:
                    S.status.done = False
                    # multi-level or single-level?
                    if len(S.levels) > 1:
                        S.status.stage = 'IT_UP'
                    else:
                        S.status.stage = 'IT_FINE'

            else:
                # if everyone is ready, end
                for S in MS:
                    S.levels[0].sweep.compute_end_point()
                    S.levels[0].hooks.dump_step(S.status)
                    S.status.stage = 'DONE'

            return MS


        if case('IT_UP'):
            # go up the hierarchy from finest to coarsest level (parallel)

            for S in MS:

                S.transfer(source=S.levels[0],target=S.levels[1])

                # sweep and send on middle levels (not on finest, not on coarsest, though)
                for l in range(1,len(S.levels)-1):
                    S.levels[l].sweep.update_nodes()
                    S.levels[l].sweep.compute_residual()
                    S.levels[l].hooks.dump_sweep(S.status)

                    if S.params.fine_comm:
                        send(S.levels[l],tag=(l,S.status.iter,S.status.slot))

                    # transfer further up the hierarchy
                    S.transfer(source=S.levels[l],target=S.levels[l+1])

                # update stage
                S.status.stage = 'IT_COARSE'

            return MS


        if case('IT_COARSE'):
            # sweeps on coarsest level (serial/blocking)

            for S in MS:

                # receive from previous step (if not first)
                if not S.status.first:
                    recv(S.levels[-1],S.prev.levels[-1],tag=(len(S.levels),S.status.iter,S.prev.status.slot))

                # do the sweep
                S.levels[-1].sweep.update_nodes()
                S.levels[-1].sweep.compute_residual()
                S.levels[-1].hooks.dump_sweep(S.status)

                # send to next step
                send(S.levels[-1],tag=(len(S.levels),S.status.iter,S.status.slot))

                # update stage
                S.status.stage = 'IT_DOWN'

            # return
            return MS


        if case('IT_DOWN'):
            # prolong corrections down to finest level (parallel)

            for S in MS:

                # receive and sweep on middle levels (except for coarsest level)
                for l in range(len(S.levels)-1,0,-1):

                    # # receive values from IT_UP (non-blocking)
                    if S.params.fine_comm and not S.status.first:
                        recv(S.levels[l-1],S.prev.levels[l-1],tag=(l-1,S.status.iter,S.prev.status.slot))

                    # prolong values
                    S.transfer(source=S.levels[l],target=S.levels[l-1])

                    # on middle levels: do sweep as usual
                    if l-1 > 0:
                        S.levels[l-1].sweep.update_nodes()
                        S.levels[l-1].sweep.compute_residual()
                        S.levels[l-1].hooks.dump_sweep(S.status)

                # update stage
                S.status.stage = 'IT_FINE'

            return MS



        #fixme: use meaningful error object here
        print('Something is wrong here, you should have hit one case statement!')
        exit()
    #fixme: use meaningful error object here
    print('Something is wrong here, you should have hit one case statement!')
    exit()




