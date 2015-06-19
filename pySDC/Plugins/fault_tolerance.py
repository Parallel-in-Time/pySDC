import copy as cp
import random as rd
import struct

iter = None
step = None
strategy = None
hard_random = 0.0

do_soft_faults = True
soft_random = 0.0
soft_counter = 0
soft_stats = []


def soft_fault_injection(nvars):
    global soft_random, soft_counter, soft_stats, do_soft_faults

    if not do_soft_faults:

        return -1,None,None,None

    else:

        doit = rd.random() < soft_random

        if doit:

            soft_counter += 1
            index = rd.randrange(nvars)
            pos = rd.randrange(31)
            uf = rd.randrange(2)
            # index = 93
            # pos = 26
            # uf = 1

            return 0,index,pos,uf

        else:

            return 0,None,None,None


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



def hard_fault_injection(S):
    global iter, step, strategy, hard_random

    if S.status.iter == 1:
        rd.seed(S.status.step)

    doit = rd.random() < hard_random
    # print(S.status.step,S.status.iter,doit)

    if ((step == S.status.step and iter == S.status.iter) or doit) and strategy is not 'NOFAULT':

        print('things went wrong here: step %i -- iteration %i -- time %e' %(S.status.step,S.status.iter,S.status.time))
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

        # S = hard_fault_correction_interp_fine(S)
        # S = hard_fault_correction_interp_coarse(S)

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



