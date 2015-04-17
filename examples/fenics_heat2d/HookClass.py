from __future__ import division
from pySDC.Hooks import hooks
from pySDC.Stats import stats

import numpy as np

class error_output(hooks):

    def __init__(self):
        """
        Initialization of particles output
        """
        super(error_output,self).__init__()


    def dump_iteration(self,status):

        super(error_output,self).dump_iteration(status)

        L = self.level
        P = L.prob

        # compute exact solution and compare
        uex = P.u_exact(status.time+status.dt)

        print('error at time %s: %s' %(status.time+status.dt,abs(uex-L.u[-1])/abs(uex)))

        return None
