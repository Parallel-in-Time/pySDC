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


    def post_iteration(self, status):

        super(error_output,self).post_iteration(status)

        L = self.level
        P = L.prob

        L.sweep.compute_end_point()

        uend = L.uend
        uex = P.u_exact(status.time+status.dt)

        err = np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(uex.values,np.inf)

        # print('error at time %s, iteration %s: %s' %(status.time,status.iter,err))

        stats.add_to_stats(step=status.step, time=status.time, iter=status.iter, type='error', value=err)
