import numpy as np


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




def run_pfasst_serial(MS,u0,t0,dt,Tend):

    num_procs = len(MS)
    slots = [p for p in range(num_procs)]

    for p in range(num_procs):

        MS[p].dt = dt
        MS[p].time = t0 + p*MS[p].dt
        MS[p].slot = slots[p]
        MS[p].init_step(u0)
        MS[p].state = 'SPREAD'
        MS[p].levels[0].stats.add_iter_stats()
        if p > 0:
            MS[p].prev = MS[slots[p-1]]

    active = [MS[p].time < Tend for p in slots]

    while any(active):

        for p in np.extract(active,slots):
            print(p,MS[p].state)
            MS = pfasst_serial(MS,p,slots)
        # This is only for ring-parallelization
        # indx = np.argsort([MS[p].time for p in slots])
        # slots = slots[indx]

        active = [MS[p].time < Tend for p in slots]

    return MS[-1].levels[0].uend



def pfasst_serial(MS,p,slots):

    S = MS[p]

    for case in switch(S.state):

        if case('SPREAD'):

            S.levels[0].sweep.predict()
            S.state = 'PREDICT_RESTRICT'
            return MS

        if case('PREDICT_RESTRICT'):

            for l in range(1,len(S.levels)):
                S.transfer(source=S.levels[l-1],target=S.levels[l])
            S.state = 'PREDICT_SWEEP'
            return MS

        if case('PREDICT_SWEEP'):

            if slots.index(p) > 0:
                prev = MS[p].prev.levels[-1].uend
                S.init_step(prev)

            S.levels[-1].sweep.update_nodes()
            S.levels[-1].sweep.compute_end_point()

            #line 63 of pfasst_flex.m comes here
            exit()


            S.state = 'PREDICT_SWEEP'
            return MS



