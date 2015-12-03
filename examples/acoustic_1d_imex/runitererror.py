
from pySDC import CollocationClasses as collclass

import numpy as np

from ProblemClass_conv import acoustic_1d_imex
from examples.acoustic_1d_imex.HookClass import plot_solution

import pySDC.Stats as st
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order
import pySDC.PFASST_stepwise as mp
from pySDC import Log
from pySDC.Stats import grep_stats, sort_stats

from matplotlib import pyplot as plt


if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 1

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-14

    sparams = {}
  
    cs_v = [0.5, 1.0, 1.5]
    sparams['maxiter'] = 25
    nodes_v = [2, 3]
      
    residual = np.zeros((np.size(cs_v), np.size(nodes_v), sparams['maxiter'])) 
    
    for cs_ind in np.arange(np.size(cs_v)):

      # This comes as read-in for the problem class
      pparams = {}
      pparams['nvars']     = [(2,250)]
      pparams['cadv']      = 0.05
      pparams['cs']        = cs_v[cs_ind]
      pparams['order_adv'] = 5
      pparams['waveno']    = 1

      # This comes as read-in for the transfer operations
      tparams = {}
      tparams['finter'] = True

      # Fill description dictionary for easy hierarchy creation
      description = {}
      description['problem_class']     = acoustic_1d_imex
      description['problem_params']    = pparams
      description['dtype_u']           = mesh
      description['dtype_f']           = rhs_imex_mesh
      description['collocation_class'] = collclass.CollGaussLobatto
      description['sweeper_class']     = imex_1st_order
      description['level_params']      = lparams
      description['hook_class']        = plot_solution
      #description['transfer_class'] = mesh_to_mesh_1d
      #description['transfer_params'] = tparams
      
      for nodes_ind in np.arange(np.size(nodes_v)):
        # setup parameters "in time"
        t0   = 0
        Tend = 0.05
        description['num_nodes'] = nodes_v[nodes_ind]

         # quickly generate block of steps
        MS = mp.generate_steps(num_procs,sparams,description)
      
        dt = Tend

        # get initial values on finest level
        P = MS[0].levels[0].prob
        uinit = P.u_exact(t0)

        # call main function to get things done...
        uend,stats = mp.run_pfasst(MS, u0=uinit, t0=t0, dt=dt, Tend=Tend)

        for k,v in stats.items():
            iter = getattr(k,'iter')
            if iter is not -1:
                residual[cs_ind, nodes_ind, iter-1] = v     
#        print residual


