import copy as cp
import random as rd

import numpy as np

# dirty, but easiest: global variables to control the injection and recovery
hard_iter = None
hard_step = None
strategy = None
hard_random = 0.0
hard_stats = []
refdata = None


def hard_fault_injection(S):
    """
    Injects a node failure and recovers using a defined strategy, can be called in the controller

    Args:
        S: the current step data structure
    Returns:
         the step after a node failure

    """

    # name global variables for this routine
    global hard_iter, hard_step, strategy, hard_stats, hard_random, refdata

    # set the seed in the first iteration, using the process number for reproducibility
    if S.status.iter == 1:
        rd.seed(S.status.slot)

    # draw random number and check if we are below our threshold (hard_random gives percentage)
    if strategy == 'NOFAULT':
        doit = rd.random() < hard_random
        if doit:
            hard_stats.append((S.status.slot, S.status.iter, S.time))
    else:
        if refdata is not None:
            # noinspection PyTypeChecker
            doit = np.any(np.all([S.status.slot, S.status.iter, S.time] == refdata, axis=1))
        else:
            doit = False

    # if we set step and iter, inject and recover (if faults are supposed to occur)
    if ((hard_step == S.status.slot and hard_iter == S.status.iter) or doit) and strategy != 'NOFAULT':

        print('things went wrong here: step %i -- iteration %i -- time %e' %
              (S.status.slot, S.status.iter, S.time))

        # add incident to statistics data type
        hard_stats.append((S.status.slot, S.status.iter, S.time))

        # ok, that's a little bit of cheating... we would need to retrieve the current residual and iteration count
        # from the previous process, but this does not matter here
        res = cp.deepcopy(S.levels[-1].status.residual)
        niter = cp.deepcopy(S.status.iter) - 1
        time = cp.deepcopy(S.time)

        # fault injection, set everything to zero or null or whatever
        S.reset_step()

        for lvl in S.levels:
            lvl.status.time = time

        # recovery
        if strategy == 'SPREAD':
            S = hard_fault_correction_spread(S)
        elif strategy == 'INTERP':
            S = hard_fault_correction_interp(S)
        elif strategy == 'INTERP_PREDICT':
            S = hard_fault_correction_interp_predict(S, res, niter)
        elif strategy == 'SPREAD_PREDICT':
            S = hard_fault_correction_spread_predict(S, res, niter)
        else:
            raise NotImplementedError('recovery strategy not implemented')

    return S


# Here come the recovery strategies

def hard_fault_correction_spread(S):
    """
        do nothing, just get new initial conditions and do sweep predict (copy)
        strategy '1-sided'

        Args:
            S: the current step (no data available)
        Returns:
            S: recovered step
    """

    # get new initial data, either from previous processes or "from scratch"
    if not S.status.first:
        ufirst = S.prev.levels[0].prob.dtype_u(S.prev.levels[0].uend)
    else:
        ufirst = S.levels[0].prob.u_exact(S.time)

    L = S.levels[0]

    # set data
    L.u[0] = L.prob.dtype_u(ufirst)
    # call prediction of the sweeper (copy the values to all nodes)
    L.sweep.predict()
    # compute uend
    L.sweep.compute_end_point()

    # proceed with fine sweep
    S.status.stage = 'IT_FINE_SWEEP'

    return S


def hard_fault_correction_interp(S):
    """
        get new initial conditions from left and uend from right, then interpolate
        strategy '2-sided'

        Args:
            S: the current step (no data available)
        Returns:
            S: recovered step
    """

    # get new initial data, either from previous processes or "from scratch"
    if not S.status.first:
        ufirst = S.prev.levels[0].prob.dtype_u(S.prev.levels[0].uend)
    else:
        ufirst = S.levels[0].prob.u_exact(S.time)

    # if I'm not the last one, get uend from following process
    # otherwise set uend = u0, so that interpolation is a copy
    if not S.status.last:
        ulast = S.next.levels[0].prob.dtype_u(S.next.levels[0].u[0])
    else:
        ulast = ufirst

    L = S.levels[0]

    # set u0, and interpolate the rest
    # evaluate f for each node (fixme: we could try to interpolate here as well)
    L.u[0] = L.prob.dtype_u(ufirst)
    L.f[0] = L.prob.eval_f(L.u[0], L.time)
    for m in range(1, L.sweep.coll.num_nodes + 1):
        L.u[m] = (1 - L.sweep.coll.nodes[m - 1]) * ufirst + L.sweep.coll.nodes[m - 1] * ulast
        L.f[m] = L.prob.eval_f(L.u[m], L.time + L.dt * L.sweep.coll.nodes[m - 1])

    # set fine level to active
    L.status.unlocked = True
    # compute uend
    L.sweep.compute_end_point()

    # proceed with fine sweep
    S.status.stage = 'IT_FINE_SWEEP'

    return S


def hard_fault_correction_spread_predict(S, res, niter):
    """
        get new initial conditions from left, copy data to nodes and correct on coarse level
        strategy '1-sided+corr'

        Args:
            S: the current step (no data available)
            res: the target residual
            niter: the max. number of iteration
        Returns:
            S: recovered step
    """

    # get new initial data, either from previous processes or "from scratch"
    if not S.status.first:
        ufirst = S.prev.levels[0].prob.dtype_u(S.prev.levels[0].uend)
    else:
        ufirst = S.levels[0].prob.u_exact(S.time)

    L = S.levels[0]

    # set u0, and copy
    L.u[0] = L.prob.dtype_u(ufirst)
    L.sweep.predict()

    # transfer to the coarsest level (overwrite values)
    for l in range(1, len(S.levels)):
        S.transfer(source=S.levels[l - 1], target=S.levels[l])

    # compute preliminary residual (just to set it)
    S.levels[-1].status.updated = True
    S.levels[-1].sweep.compute_residual()
    # keep sweeping until either k < niter or the current residual is lower than res (niter, res was stored before
    # fault injection (lazy, should get this from the previous process)
    k = 0
    if res is not None:
        while S.levels[-1].status.residual > res and k < niter:
            k += 1
            print(S.levels[-1].status.residual, res, k)
            S.levels[-1].sweep.update_nodes()
            S.levels[-1].sweep.compute_residual()

    # transfer back to finest level (coarse correction!)
    for l in range(len(S.levels) - 1, 0, -1):
        S.transfer(source=S.levels[l], target=S.levels[l - 1])

    # compute uend
    L.sweep.compute_end_point()

    # proceed with fine sweep
    S.status.stage = 'IT_FINE_SWEEP'

    return S


def hard_fault_correction_interp_predict(S, res, niter):
    """
        get new initial conditions from left and uend from right, interpolate data to nodes and correct on coarse level
        strategy '2-sided+corr'

        Args:
            S: the current step (no data available)
            res: the target residual
            niter: the max. number of iteration
        Returns:
            S: recovered step
    """

    # get new initial data, either from previous processes or "from scratch"
    if not S.status.first:
        ufirst = S.prev.levels[0].prob.dtype_u(S.prev.levels[0].uend)
    else:
        ufirst = S.levels[0].prob.u_exact(S.time)

    # if I'm not the last one, get uend from following process
    # otherwise set uend = u0, so that interpolation is a copy
    if not S.status.last:
        ulast = S.next.levels[0].prob.dtype_u(S.next.levels[0].u[0])
    else:
        ulast = ufirst

    L = S.levels[0]

    # set u0, and interpolate the rest
    # evaluate f for each node (fixme: we could try to interpolate here as well)
    L.u[0] = L.prob.dtype_u(ufirst)
    L.f[0] = L.prob.eval_f(L.u[0], L.time)
    for m in range(1, L.sweep.coll.num_nodes + 1):
        L.u[m] = (1 - L.sweep.coll.nodes[m - 1]) * ufirst + L.sweep.coll.nodes[m - 1] * ulast
        L.f[m] = L.prob.eval_f(L.u[m], L.time + L.dt * L.sweep.coll.nodes[m - 1])

    # set fine level to active
    L.status.unlocked = True

    # transfer to the coarsest level (overwrite values)
    for l in range(1, len(S.levels)):
        S.transfer(source=S.levels[l - 1], target=S.levels[l])

    # compute preliminary residual (just to set it)
    S.levels[-1].status.updated = True
    S.levels[-1].sweep.compute_residual()
    # keep sweeping until either k < niter or the current residual is lower than res (niter, res was stored before
    # fault injection (lazy, should get this from the previous process)
    k = 0
    if res is not None:
        while S.levels[-1].status.residual > res and k < niter:
            k += 1
            print(S.levels[-1].status.residual, res, k)
            S.levels[-1].sweep.update_nodes()
            S.levels[-1].sweep.compute_residual()

    # transfer back to finest level (coarse correction!)
    for l in range(len(S.levels) - 1, 0, -1):
        S.transfer(source=S.levels[l], target=S.levels[l - 1])

    # compute uend
    L.sweep.compute_end_point()

    # proceed with fine sweep
    S.status.stage = 'IT_FINE_SWEEP'

    return S
