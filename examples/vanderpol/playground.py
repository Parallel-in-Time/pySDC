from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pySDC.deprecated.PFASST_stepwise as mp
from implementations.datatype_classes import mesh
from implementations.sweeper_classes.linearized_implicit_fixed_parallel import linearized_implicit_fixed_parallel
from pySDC.Stats import grep_stats, sort_stats

from examples.vanderpol.HookClass import vanderpol_output
from pySDC import CollocationClasses as collclass
from pySDC import Log
from pySDC.implementations.problem_classes.Van_der_Pol_oscillator import vanderpol
from pySDC.plugins.sweeper_helper import get_Qd

if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 1

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-10

    sparams = {}
    sparams['maxiter'] = 20

    # This comes as read-in for the problem class
    pparams = {}
    pparams['newton_tol'] = 1E-12
    pparams['maxiter'] = 50
    pparams['mu'] = 5
    pparams['u0'] = np.array([2.0,0])

    Nnodes = 5
    cclass = collclass.CollGaussRadau_Right

    swparams = {}
    swparams['QI'] = get_Qd(cclass, qd_type='IEpar')
    swparams['fixed_time_in_jacobian'] = 0

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = vanderpol
    description['problem_params'] = pparams
    description['dtype_u'] = mesh
    description['dtype_f'] = mesh
    description['collocation_class'] = cclass
    description['num_nodes'] = Nnodes
    # description['sweeper_class'] = generic_LU
    # description['sweeper_class'] = linearized_implicit_parallel
    description['sweeper_class'] = linearized_implicit_fixed_parallel
    description['sweeper_params'] = swparams
    description['level_params'] = lparams
    description['hook_class'] = vanderpol_output # this is optional

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs,sparams,description)

    # setup parameters "in time"
    t0 = 0
    dt = 20*0.1
    Tend = 2

    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    print('Init:',uinit.values)

    # call main function to get things done...
    uend,stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    # get stats and error
    extract_stats = grep_stats(stats,type='niter')
    sortedlist_stats = sort_stats(extract_stats,sortby='time')

    np.set_printoptions(16)
    print('u_end:',uend.values,' at time',Tend)

    # this is for Tend = 2.0, computed with 2k time-steps and M=3 (G-Le)
    if Tend == 2.0:
        uex = np.array([1.7092338721248415, -0.17438654047532 ])
        print('Error:',np.linalg.norm(uex-uend.values,np.inf)/np.linalg.norm(uex,np.inf))

    plt.show()
