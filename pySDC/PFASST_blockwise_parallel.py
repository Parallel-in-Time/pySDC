import itertools
import copy as cp
import numpy as np

from pySDC.Stats import stats

from pySDC.PFASST_helper import *


def run_pfasst(S,u0,t0,dt,Tend,comm):
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

    rank = comm.Get_rank()
    num_procs = comm.Get_size()

    all_dt = comm.allgather(dt)

    S.status.dt = dt
    S.status.time = t0 + sum(all_dt[0:rank])
    S.status.step = rank
    S.status.slot = rank

    active = S.status.time < Tend - 10*np.finfo(float).eps

    uend = u0
    while active:
        # initialize block of steps with u0
        S = restart_block(S,num_procs,uend)
        # call pre-start hook
        S.levels[0].hooks.dump_pre(S.status)
        all_done = comm.allgather(S.status.done)

        while not all(all_done):
            S = pfasst(S,comm)
            all_done = comm.allgather(S.status.done)

        if all(all_done):
            tend = comm.bcast(S.status.time, root=num_procs-1)
            stepend = comm.bcast(S.status.step, root=num_procs-1)
            all_dt = comm.allgather(dt)
            S.status.time = tend + all_dt[-1] + sum(all_dt[0:rank])
            S.status.step = stepend + rank + 1
            uend = comm.bcast(S.levels[0].uend, root=num_procs-1)
            active = S.status.time < Tend - 10 * np.finfo(float).eps

    return uend,stats.return_stats()


def restart_block(S,size,u0):
    """
    Helper routine to reset/restart block of (active) steps

    Args:
        MS: block of (all) steps
        active_slots: list of active steps
        u0: initial value to distribute across the steps

    Returns:
        block of (all) steps
    """

    # store link to previous step
    S.prev = S.status.slot - 1
    S.next = S.status.slot + 1
    # resets step
    S.reset_step()
    # determine whether I am the first and/or last in line
    S.status.first = S.prev == -1
    S.status.last = S.next == size
    # intialize step with u0
    S.init_step(u0)
    # reset some values
    S.status.done = False
    S.status.iter = 0
    S.status.stage = 'SPREAD'
    for l in S.levels:
        l.tag = None

    return S


def recv(target,source,tag,comm):
    """
    Receive function

    Args:
        target: level which will receive the values
        source: level which initiated the send
        tag: identifier to check if this message is really for me
    """

    target.u[0] = comm.recv(source=source, tag=tag)
    target.f[0] = target.prob.eval_f(target.u[0], target.time)


def send(source,target,tag,comm):
    """
    Send function

    Args:
        source: level which has the new values
        tag: identifier for this message
    """
    # sending here means computing uend ("one-sided communication")
    source.sweep.compute_end_point()
    comm.send(source.uend, dest = target, tag = tag)


def predictor(S, comm):
    """
    Predictor function, extracted from the stepwise implementation (will be also used by matrix sweppers)

    Args:
        MS: multiple steps

    Returns:
        block of steps with initial values
    """

    # restrict to coarsest level
    for l in range(1, len(S.levels)):
        S.transfer(source=S.levels[l-1],target=S.levels[l])


    for p in range(S.status.slot+1):

        if not p == 0 and not S.status.first:
            recv(target=S.levels[-1], source=S.prev, tag=S.status.iter, comm=comm)

        # do the sweep with new values
        S.levels[-1].sweep.update_nodes()

        if not S.status.last:
            send(source=S.levels[-1], target=S.next, tag=S.status.iter, comm=comm)

    # interpolate back to finest level
    for l in range(len(S.levels)-1,0,-1):
        S.transfer(source=S.levels[l],target=S.levels[l-1])

    return S


def pfasst(S,comm):
    """
    Main function including the stages of SDC, MLSDC and PFASST (the "controller")

    For the workflow of this controller, check out one of our PFASST talks

    Args:
        MS: all active steps

    Returns:
        all active steps
    """

    # if all stages are the same, continue, otherwise abort
    all_stage = comm.allgather(S.status.stage)

    if not all(all_stage):
        print('not all stages are equal, aborting..')
        exit()

    # print(S.status.slot,stage)

    for case in switch(S.status.stage):

        if case('SPREAD'):
            # (potentially) serial spreading phase

            # first stage: spread values
            S.levels[0].hooks.pre_step(S.status)

            # call predictor from sweeper
            S.levels[0].sweep.predict()

            # update stage
            if len(S.levels) > 1 and S.params.predict:
                S.status.stage = 'PREDICT'
            else:
                S.levels[0].hooks.dump_pre_iteration(S.status)
                S.status.stage = 'IT_FINE'
            return S


        if case('PREDICT'):
            # call predictor (serial)

            S = predictor(S, comm)

            # update stage
            S.levels[0].hooks.dump_pre_iteration(S.status)
            S.status.stage = 'IT_FINE'

            return S


        if case('IT_FINE'):
            # do fine sweep

            # increment iteration count here (and only here)
            S.status.iter += 1

            # standard sweep workflow: update nodes, compute residual, log progress
            S.levels[0].sweep.update_nodes()
            S.levels[0].sweep.compute_residual()
            S.levels[0].hooks.dump_sweep(S.status)

            S.levels[0].hooks.dump_iteration(S.status)

            # update stage
            S.status.stage = 'IT_CHECK'

            return S


        if case('IT_CHECK'):
            # check whether to stop iterating (parallel)

            S.status.done = check_convergence(S)
            all_done = comm.allgather(S.status.done)

            # if not everyone is ready yet, keep doing stuff
            if not all(all_done):
                S.status.done = False
                # multi-level or single-level?
                if len(S.levels) > 1:
                    S.status.stage = 'IT_UP'
                else:
                    S.status.stage = 'IT_FINE'

            else:
                # S.levels[0].sweep.compute_end_point()
                S.levels[0].hooks.dump_step(S.status)
                S.status.stage = 'DONE'

            return S


        if case('IT_UP'):
            # go up the hierarchy from finest to coarsest level (parallel)


            S.transfer(source=S.levels[0],target=S.levels[1])

            # sweep and send on middle levels (not on finest, not on coarsest, though)
            for l in range(1,len(S.levels)-1):
                S.levels[l].sweep.update_nodes()
                S.levels[l].sweep.compute_residual()
                S.levels[l].hooks.dump_sweep(S.status)

                # transfer further up the hierarchy
                S.transfer(source=S.levels[l],target=S.levels[l+1])

            # update stage
            S.status.stage = 'IT_COARSE'

            return S


        if case('IT_COARSE'):
            # sweeps on coarsest level (serial/blocking)


            # receive from previous step (if not first)
            if not S.status.first:
                recv(target=S.levels[-1], source=S.prev, tag=S.status.iter, comm=comm)

            # do the sweep
            S.levels[-1].sweep.update_nodes()
            S.levels[-1].sweep.compute_residual()
            S.levels[-1].hooks.dump_sweep(S.status)
            S.levels[-1].sweep.compute_end_point()

            # send to next step
            if not S.status.last:
                send(source=S.levels[-1], target=S.next, tag=S.status.iter, comm=comm)

            # update stage
            S.status.stage = 'IT_DOWN'

            # return
            return S


        if case('IT_DOWN'):
            # prolong corrections down to finest level (parallel)

            # receive and sweep on middle levels (except for coarsest level)
            for l in range(len(S.levels)-1,0,-1):

                # prolong values
                S.transfer(source=S.levels[l],target=S.levels[l-1])
                S.levels[l-1].sweep.compute_end_point()

                if not S.status.last and S.params.fine_comm:
                    req_send = comm.isend(S.levels[l-1].uend,dest=S.next,tag=S.status.iter)

                if not S.status.first and S.params.fine_comm:
                    recv(target=S.levels[l-1], source=S.prev, tag=S.status.iter, comm=comm)

                if not S.status.last and S.params.fine_comm:
                    req_send.wait()

                # on middle levels: do sweep as usual
                if l-1 > 0:
                    S.levels[l-1].sweep.update_nodes()
                    S.levels[l-1].sweep.compute_residual()
                    S.levels[l-1].hooks.dump_sweep(S.status)

            # update stage
            S.status.stage = 'IT_FINE'

            return S



        #fixme: use meaningful error object here
        print('Something is wrong here, you should have hit one case statement!')
        exit()
    #fixme: use meaningful error object here
    print('Something is wrong here, you should have hit one case statement!')
    exit()




