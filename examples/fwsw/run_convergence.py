from __future__ import print_function

from pySDC import CollocationClasses as collclass

import numpy as np
import matplotlib.pyplot as plt

from examples.fwsw.ProblemClass import swfw_scalar
from pySDC.datatype_classes.complex_mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order
import pySDC.PFASST_stepwise as mp
from pySDC import Log



if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 1

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-12

    sparams = {}
    sparams['maxiter'] = 4

    # This comes as read-in for the problem class
    pparams = {}
    pparams['lambda_s'] = np.array([0.1j], dtype='complex')
    pparams['lambda_f'] = np.array([1.0j], dtype='complex')
    pparams['u0'] = 1

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = swfw_scalar
    description['problem_params'] = pparams
    description['dtype_u'] = mesh
    description['dtype_f'] = rhs_imex_mesh
    description['collocation_class'] = collclass.CollGaussLegendre
    description['num_nodes'] = [3]
    description['do_LU'] = False
    description['sweeper_class'] = imex_1st_order
    description['level_params'] = lparams

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs,sparams,description)

    Nsteps_v = np.array([1, 2, 4, 8, 10, 15, 20])
    Tend = 1.0
    t0   = 0

    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    uex = P.u_exact(Tend)
    error = np.zeros(np.size(Nsteps_v))
    convline = np.zeros(np.size(Nsteps_v))
    
    for j in range(0,np.size(Nsteps_v)):
    # setup parameters "in time"
      dt   = Tend/float(Nsteps_v[j])

      # call main function to get things done...
      uend,stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)
      error[j] = np.abs(uend.values - uex.values)
      convline[j] = error[j]*(float(Nsteps_v[j])/float(Nsteps_v[j]))**sparams['maxiter']

    plt.figure()
    plt.loglog(Nsteps_v, error, 'bo', markersize=12)
    plt.loglog(Nsteps_v, convline, '-', color='k')
    plt.show()