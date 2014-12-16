import numpy as np
import copy as cp


class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False


def check_convergence(S):

        L = S.levels[0]

        res = L.status.residual

        converged = S.iter >= S.params.maxiter or res <= L.params.restol

        L.stats.iter_stats[-1].residual = res

        if converged:
            S.stats.niter = S.iter
            L.stats.residual = res
            S.stats.residual = res

        return converged


def run_pfasst_serial(MS,u0,t0,dt,Tend):

    num_procs = len(MS)
    slots = [p for p in range(num_procs)]

    for p in range(num_procs):

        MS[p].dt = dt
        MS[p].time = t0 + p*MS[p].dt
        MS[p].slot = slots[p]
        MS[p].init_step(u0)
        MS[p].stage = 'SPREAD'
        MS[p].levels[0].stats.add_iter_stats()
        MS[p].prev = MS[slots[p-1]]
        MS[p].pred_cnt = -1
        MS[p].iter = 0
        MS[p].done = False

    active = [MS[p].time < Tend for p in slots]

    while any(active):

        for p in np.extract(active,slots):
            print(p,MS[p].stage)
            MS = pfasst_serial(MS,p,slots)
        # This is only for ring-parallelization
        # indx = np.argsort([MS[p].time for p in slots])
        # slots = slots[indx]

        active = [MS[p].time < Tend for p in slots]

    return MS[-1].levels[0].uend



def pfasst_serial(MS,p,slots):

    S = MS[p]

    for case in switch(S.stage):

        if case('SPREAD'):

            S.levels[0].sweep.predict()
            S.stage = 'PREDICT_RESTRICT'
            return MS

        if case('PREDICT_RESTRICT'):

            for l in range(1,len(S.levels)):
                S.transfer(source=S.levels[l-1],target=S.levels[l])
            S.stage = 'PREDICT_SWEEP'
            return MS

        if case('PREDICT_SWEEP'):

            recv(S.levels[-1],S.prev.levels[-1],slots,p)

            S.levels[-1].sweep.update_nodes()
            S.levels[-1].sweep.compute_end_point()

            S.pred_cnt += 1
            if S.pred_cnt == slots.index(p):
                S.stage = 'PREDICT_INTERP'
            else:
                S.stage = 'PREDICT_SWEEP'

            return MS

        if case('PREDICT_INTERP'):

            for l in range(len(S.levels)-1,0,-1):
                S.transfer(source=S.levels[l],target=S.levels[l-1])
            S.stage = 'IT_SWEEP_FINE'
            return MS

        if case('IT_SWEEP_FINE'):

            S.iter += 1
            S.levels[0].sweep.update_nodes()
            S.levels[0].sweep.compute_end_point()
            S.stage = 'IT_CHECK'
            return MS

        if case('IT_CHECK'):

            S.levels[0].sweep.compute_residual()

            S.done = check_convergence(S)

            if S.done and not S.prev.done:
                S.done = False

            if S.done:
                S.stage = 'DONE'
            else:
                S.stage = 'IT_UP'

            return MS





def recv(target,source,slots,p):

    if slots.index(p) > 0:
        target.u[0] = cp.deepcopy(source.uend)






