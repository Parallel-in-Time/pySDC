import copy as cp
import random as rd
import struct

hard_iter = None
hard_step = None
strategy = None
hard_random = 0.0
hard_stats = []

soft_do_faults = False

soft_step = 0
soft_iter = 0
soft_level = 0
soft_node = 0

soft_fault_injected = 0
soft_fault_detected = 0
soft_fault_missed = 0
soft_fault_hit = 0

soft_safety_factor = 1.0

soft_stats = []
soft_do_correction = False


def soft_fault_preproc(nsteps,niters,nlevs,nnodes):
    global soft_step, soft_iter, soft_level, soft_node

    rd.seed()

    soft_step = rd.randrange(nsteps)
    soft_iter = rd.randrange(1,niters+1)
    soft_level = rd.randrange(nlevs)
    # FIXME: do we need to exclude the first node??
    soft_node = rd.randrange(1,nnodes)
    # print(soft_step,soft_iter,soft_level,soft_node)

def soft_fault_injection(step,iter,level,node,nvars):
    global soft_step, soft_iter, soft_level, soft_node, soft_do_faults, soft_stats

    doit = step == soft_step and iter == soft_iter and level == soft_level and node == soft_node and soft_do_faults

    if doit:
        rd.seed()
        index = rd.randrange(nvars)
        pos = rd.randrange(31)
        uf = rd.randrange(2)

        return index,pos,uf

    else:

        return None,None,None


def __bitsToFloat(b):
    s = struct.pack('>l', b)
    return struct.unpack('>f', s)[0]

def __floatToBits(f):
    s = struct.pack('>f', f)
    return struct.unpack('>l', s)[0]

def do_bitflip(a,pos=29):

    b = __floatToBits(a)
    mask = 1<<pos
    c = b^mask

    return __bitsToFloat(c)


# def hard_fault_preproc(time,nsteps,MS):
#     global hard_step, hard_iter
#
#     rd.seed(time)
#
#     if nsteps > 0:
#         hard_step = rd.randrange(nsteps)
#     else:
#         hard_step = 0
#
#     if MS[hard_step].status.iter is not None:
#         niters = MS[hard_step].status.iter
#     else:
#         niters = 8
#
#     hard_iter = rd.randrange(1,niters+1)
#     print(hard_step,hard_iter)


def hard_fault_injection(S):
    global hard_iter, hard_step, strategy, hard_stats, hard_random

    if S.status.iter == 1:
        rd.seed(S.status.step)

    doit = rd.random() < hard_random and S.status.iter < 8

    # print(S.status.step,S.status.iter,hard_step,hard_iter)

    if ((hard_step == S.status.step and hard_iter == S.status.iter) or doit) and strategy is not 'NOFAULT':

        print('things went wrong here: step %i -- iteration %i -- time %e' %(S.status.step,S.status.iter,S.status.time))

        hard_stats.append((S.status.step,S.status.iter,S.status.time))

        res = cp.deepcopy(S.levels[-1].status.residual)
        niter = cp.deepcopy(S.status.iter)-1

        S.reset_step()

        if strategy is 'SPREAD':
            S = hard_fault_correction_spread(S)
        elif strategy is 'INTERP':
            S = hard_fault_correction_interp_all(S)
        elif strategy is 'INTERP_PREDICT':
            S = hard_fault_correction_predict(S,res,niter)
        elif strategy is 'SPREAD_PREDICT':
            S = hard_fault_correction_spread_predict(S,res,niter)
        else:
            print('strategy not implemented, aborting...',strategy)
            exit()

    return S


def hard_fault_correction_spread(S):

    for l in range(len(S.levels)):

        if not S.status.first:
            ufirst = S.prev.levels[l].prob.dtype_u(S.prev.levels[l].uend)
        else:
            ufirst = S.levels[l].prob.u_exact(S.status.time)

        S.levels[l].u[0] = ufirst
        S.levels[l].sweep.predict()
        S.levels[l].sweep.compute_end_point()


    S.status.stage = 'IT_FINE_SWEEP'

    return S


def hard_fault_correction_interp_all(S):

    for l in range(len(S.levels)):

        if not S.status.first:
            ufirst = S.prev.levels[l].prob.dtype_u(S.prev.levels[l].uend)
        else:
            ufirst = S.levels[l].prob.u_exact(S.status.time)

        if not S.status.last:
            ulast = S.next.levels[l].prob.dtype_u(S.next.levels[l].u[0])
        else:
            ulast = ufirst

        L = S.levels[l]

        ffirst = L.prob.eval_f(ufirst,L.time)
        flast = L.prob.eval_f(ulast,L.time + L.dt)

        L.u[0] = L.prob.dtype_u(ufirst)
        L.f[0] = L.prob.eval_f(L.u[0],L.time)
        for m in range(1,L.sweep.coll.num_nodes+1):
            L.u[m] = (1-L.sweep.coll.nodes[m-1])*ufirst + L.sweep.coll.nodes[m-1]*ulast
            L.f[m] = L.prob.eval_f(L.u[m],L.time+L.dt*L.sweep.coll.nodes[m-1])

        L.status.unlocked = True

        # L.sweep.update_nodes()
        L.sweep.compute_end_point()

    return S


def hard_fault_correction_spread_predict(S,res,niter):

    if not S.status.first:
        ufirst = S.prev.levels[0].prob.dtype_u(S.prev.levels[0].uend)
    else:
        ufirst = S.levels[0].prob.u_exact(S.status.time)

    L = S.levels[0]

    L.u[0] = L.prob.dtype_u(ufirst)
    S.levels[0].sweep.predict()

    for l in range(1,len(S.levels)):
        S.transfer(source=S.levels[l-1],target=S.levels[l])

    S.levels[-1].sweep.compute_residual()
    k = 0
    if res is not None:
        while S.levels[-1].status.residual > res and k < niter:
            k += 1
            print(S.levels[-1].status.residual,res,k)
            S.levels[-1].sweep.update_nodes()
            S.levels[-1].sweep.compute_residual()

    for l in range(len(S.levels)-1,0,-1):
        S.transfer(source=S.levels[l],target=S.levels[l-1])

    S.levels[0].sweep.compute_end_point()

    return S


def hard_fault_correction_interp_coarse(S):

    if not S.status.first:
        ufirst = S.prev.levels[0].prob.dtype_u(S.prev.levels[0].uend)
    else:
        ufirst = S.levels[0].prob.u_exact(S.status.time)

    S.levels[0].u[0] = S.levels[0].prob.dtype_u(ufirst)
    S.levels[0].sweep.predict()

    S.levels[0].status.unlocked = True

    S.transfer(source=S.levels[0],target=S.levels[1])

    if not S.status.first:
        ufirst = S.prev.levels[-1].prob.dtype_u(S.prev.levels[-1].uend)
    else:
        ufirst = S.levels[-1].prob.u_exact(S.status.time)

    if not S.status.last:
        ulast = S.next.levels[-1].prob.dtype_u(S.next.levels[-1].u[0])
    else:
        ulast = ufirst

    L = S.levels[-1]

    L.u[0] = L.prob.dtype_u(ufirst)
    L.f[0] = L.prob.eval_f(L.u[0],L.time)
    for m in range(1,L.sweep.coll.num_nodes+1):
        L.u[m] = (1-L.sweep.coll.nodes[m-1])*ufirst + L.sweep.coll.nodes[m-1]*ulast
        L.f[m] = L.prob.eval_f(L.u[m],L.time+L.dt*L.sweep.coll.nodes[m-1])

    L.status.unlocked = True
    S.levels[1].sweep.compute_end_point()
    S.transfer(source=S.levels[1],target=S.levels[0])
    S.levels[0].sweep.compute_end_point()

    return S


def hard_fault_correction_predict(S,res,niter):

    if not S.status.first:
        ufirst = S.prev.levels[0].prob.dtype_u(S.prev.levels[0].uend)
    else:
        ufirst = S.levels[0].prob.u_exact(S.status.time)

    if not S.status.last:
        ulast = S.next.levels[0].prob.dtype_u(S.next.levels[0].u[0])
    else:
        ulast = ufirst

    L = S.levels[0]

    L.u[0] = L.prob.dtype_u(ufirst)
    L.f[0] = L.prob.eval_f(L.u[0],L.time)
    for m in range(1,L.sweep.coll.num_nodes+1):
        L.u[m] = (1-L.sweep.coll.nodes[m-1])*ufirst + L.sweep.coll.nodes[m-1]*ulast
        L.f[m] = L.prob.eval_f(L.u[m],L.time+L.dt*L.sweep.coll.nodes[m-1])

    L.status.unlocked = True

    for l in range(1,len(S.levels)):
        S.transfer(source=S.levels[l-1],target=S.levels[l])

    S.levels[-1].sweep.compute_residual()
    k = 0
    if res is not None:
        while S.levels[-1].status.residual > res and k < niter:
            k += 1
            print(S.levels[-1].status.residual,res,k)
            S.levels[-1].sweep.update_nodes()
            S.levels[-1].sweep.compute_residual()

    for l in range(len(S.levels)-1,0,-1):
        S.transfer(source=S.levels[l],target=S.levels[l-1])

    S.levels[0].sweep.compute_end_point()

    return S



