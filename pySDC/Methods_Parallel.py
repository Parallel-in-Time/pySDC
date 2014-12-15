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
    slots = np.array(range(num_procs))

    for p in range(num_procs):

        MS[p].dt = dt
        MS[p].time = t0 + p*MS[p].dt
        MS[p].slot = slots[p]

    MS[0].init_step(u0)
    MS[0].state = 'INIT'

    for p in range(1,num_procs):
        MS[p].state = 'STARTUP'


    active = [MS[p].time < Tend for p in range(num_procs)]

    while any(active):
        for p in range(num_procs):
            print(p)
            MS[p] = pfasst_serial(MS[p],num_procs)
        active = [MS[p].time < Tend for p in range(num_procs)]

    return MS[-1].levels[0].uend



def pfasst_serial(S,num_procs):

    for case in switch(S.state):
        if case('INIT'):
            S.time += num_procs*S.dt
            print(S.state,S.time)
            return S
        if case('STARTUP'):
            S.time += num_procs*S.dt
            print(S.state,S.time)
            return S






