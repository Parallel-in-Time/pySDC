import numpy as np
import copy as cp
import logging


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

    # fixme: deal with stats
    # fixme: add ring parallelization as before
    # fixme: encap initialization of new step
    # fixme: simplify send (and fix IT_DOWN)

    num_procs = len(MS)
    slots = [p for p in range(num_procs)]

    for p in slots:

        MS[p].dt = dt
        MS[p].time = t0 + p*MS[p].dt

    active = [MS[p].time < Tend for p in slots]
    active_slots = [p for p in slots if active[slots[p]]]

    #fixme: encap this
    for p in active_slots:

        MS[p].slot = active_slots[p]
        MS[p].prev = MS[active_slots[p-1]] # ring here for simplicity, need to test during recv whether I'm not the first
        MS[p].first = active_slots.index(p) == 0
        MS[p].last = active_slots.index(p) == len(active_slots)-1
        MS[p].init_step(u0)
        MS[p].stage = 'SPREAD'
        MS[p].levels[0].stats.add_iter_stats()
        MS[p].pred_cnt = active_slots.index(p)+1
        MS[p].iter = 0
        MS[p].done = False
        for l in MS[p].levels:
            l.tag = False



    while any(active):

        for p in active_slots:
            # print(p,MS[p].stage)
            MS = pfasst_serial(MS,p)

        finished = [MS[p].done for p in active_slots]

        # for block-parallelization
        if all(finished):

            uend = MS[active_slots[-1]].levels[0].uend # FIXME: only true for non-ring-parallelization?

            active = [MS[p].time+num_procs*MS[p].dt < Tend for p in slots]
            active_slots = [p for p in slots if active[slots[p]]]

            #fixme: encap this
            for p in active_slots:
                MS[p].first = active_slots.index(p) == 0
                MS[p].last = active_slots.index(p) == len(active_slots)-1
                MS[p].time += num_procs*MS[p].dt
                MS[p].reset_step()
                MS[p].init_step(uend)
                MS[p].done = False
                MS[p].pred_cnt = active_slots.index(p)+1 # fixme: how to do this for ring-parallelization?
                MS[p].iter = 0
                MS[p].stage = 'SPREAD'
                MS[p].levels[0].stats.add_iter_stats()
                for l in MS[p].levels:
                    l.tag = False


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

    return uend



def pfasst_serial(MS,p):

    S = MS[p]

    if S.done:
        return MS

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

            if not S.first:
                if S.prev.levels[-1].tag:
                    recv(S.levels[-1],S.prev.levels[-1])
                    S.prev.levels[-1].tag = False


            S.levels[-1].sweep.update_nodes()

            S.stage = 'PREDICT_SEND'

            return MS

        if case('PREDICT_SEND'):

            if not S.last:
                if not S.levels[-1].tag:
                    send(S.levels[-1],tag=True)
                else:
                    S.stage = 'PREDICT_SEND'
                    return MS

            S.pred_cnt -= 1
            if S.pred_cnt == 0:
                S.stage = 'PREDICT_INTERP'
            else:
                S.stage = 'PREDICT_SWEEP'

            return MS

        if case('PREDICT_INTERP'):

            for l in range(len(S.levels)-1,0,-1):
                S.transfer(source=S.levels[l],target=S.levels[l-1])
            S.stage = 'IT_FINE_SWEEP'
            return MS

        if case('IT_FINE_SWEEP'):

            S.iter += 1
            S.levels[0].sweep.update_nodes()

            S.levels[0].sweep.compute_residual()
            S.levels[0].logger.info('Process %2i at stage %s: Level: %s -- Iteration: %2i -- Residual: %12.8e',
                                    p,S.stage,S.levels[0].id,S.iter,S.levels[0].status.residual)

            S.stage = 'IT_FINE_SEND'

            return MS

        if case('IT_FINE_SEND'):

            if S.last:
                S.stage = 'IT_CHECK'
            else:
                if not S.levels[0].tag:
                    send(S.levels[0],tag=True)
                    S.stage = 'IT_CHECK'
                else:
                    S.stage = 'IT_FINE_SEND'

            return MS


        if case('IT_CHECK'):

            S.done = check_convergence(S)

            if not S.first and S.done and not S.prev.done:
                S.done = False

            if S.done:
                S.levels[0].sweep.compute_end_point()
                S.stage = 'DONE'
            else:
                S.stage = 'IT_UP'

            return MS

        if case('IT_UP'):

            S.transfer(source=S.levels[0],target=S.levels[1])

            for l in range(1,len(S.levels)-1):
                S.levels[l].sweep.update_nodes()

                if not S.last:
                    if not S.levels[l].tag:
                        send(S.levels[l],tag=True)
                    else:
                        print('SEND ERROR',l,S.slot,S.levels[l].tag)
                        exit()

                S.levels[l].sweep.compute_residual()
                S.levels[l].logger.info('Process %2i at stage %s: Level: %s -- Iteration: %2i -- Residual: %12.8e',
                                        p,S.stage,S.levels[l].id,S.iter,S.levels[l].status.residual)
                S.transfer(source=S.levels[l],target=S.levels[l+1])

            S.stage = 'IT_COARSE_SWEEP'
            return MS

        if case('IT_COARSE_SWEEP'):

            if not S.first and not S.prev.done:
                if S.prev.levels[-1].tag:
                    recv(S.levels[-1],S.prev.levels[-1])
                    S.prev.levels[-1].tag = False
                else:
                    print('RECV ERROR COARSE',S.slot,S.prev.slot,S.iter,S.prev.levels[-1].tag)
                    exit()

            S.levels[-1].sweep.update_nodes()
            S.levels[-1].sweep.compute_residual()
            S.levels[-1].logger.info('Process %2i at stage %s: Level: %s -- Iteration: %2i -- Residual: %12.8e',
                                     p,S.stage,S.levels[-1].id,S.iter,S.levels[-1].status.residual)
            S.stage = 'IT_COARSE_SEND'
            return MS


        if case('IT_COARSE_SEND'):

            if S.last:
                S.stage = 'IT_DOWN'
            else:
                if not S.levels[-1].tag:
                    send(S.levels[-1],tag=True)
                    S.stage = 'IT_DOWN'
                else:
                    S.stage = 'IT_COARSE_SEND'

            return MS


        if case('IT_DOWN'):

            for l in range(len(S.levels)-1,0,-1):

                if not S.first and not S.prev.done:
                    if S.prev.levels[l-1].tag:
                        recv(S.levels[l-1],S.prev.levels[l-1])
                        S.prev.levels[l-1].tag = False
                    else:
                        if not S.prev.done:
                            print('RECV ERROR')
                            exit()

                S.transfer(source=S.levels[l],target=S.levels[l-1])

                if l-1 > 0:
                    S.levels[l-1].sweep.update_nodes()

                    #fixme: send
                    S.levels[l-1].sweep.compute_end_point()
                    S.levels[l-1].sweep.compute_residual()
                    S.levels[l-1].logger.info('Process %2i at stage %s: Level: %s -- Iteration: %2i -- Residual: '
                                              '%12.8e', p,S.stage,S.levels[l].id,S.iter,S.levels[l].status.residual)

            S.stage = 'IT_FINE_SWEEP'
            return MS


def recv(target,source):
    target.u[0] = cp.deepcopy(source.uend)

def send(source,tag):
    source.sweep.compute_end_point()
    source.tag = cp.deepcopy(tag)




