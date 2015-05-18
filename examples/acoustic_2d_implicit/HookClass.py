from __future__ import division
from pySDC.Hooks import hooks
from pySDC.Stats import stats

import matplotlib.pyplot as plt
import numpy as np

class plot_solution(hooks):

    def __init__(self):
        """
        Initialization of output
        """
        super(plot_solution,self).__init__()

        # add figure object for further use
        fig = plt.figure(figsize=(8,8))

    def dump_step(self,status):
        """
        Overwrite standard dump per step

        Args:
            status: status object per step
        """
        super(plot_solution,self).dump_step(status)

        print "hook..."
        plt.pause(0.00001)

        return None
