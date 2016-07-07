from __future__ import print_function

import numpy as np

from pySDC import CollocationClasses as collclass
from examples.auzinger.ProblemClass import auzinger
from pySDC.datatype_classes.mesh import mesh
from pySDC.sweeper_classes.generic_LU import generic_LU
from pySDC.sweeper_classes.linearized_implicit import linearized_implicit
from pySDC.sweeper_classes.linearized_implicit_fixed import linearized_implicit_fixed
import pySDC.PFASST_stepwise as mp
from pySDC import Log
from pySDC.Stats import grep_stats, sort_stats

from pySDC.Plugins.sweeper_helper import get_Qd

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

    Nnodes = 3
    cclass = collclass.CollGaussRadau_Right

    swparams = {}
    swparams['QI'] = get_Qd(cclass, Nnodes=Nnodes, qd_type='LU')
    swparams['fixed_time_in_jacobian'] = 0

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = auzinger
    description['problem_params'] = pparams
    description['dtype_u'] = mesh
    description['dtype_f'] = mesh
    description['collocation_class'] = cclass
    description['num_nodes'] = Nnodes
    description['sweeper_class'] = generic_LU
    # description['sweeper_class'] = linearized_implicit
    # description['sweeper_class'] = linearized_implicit_fixed
    description['sweeper_params'] = swparams
    description['level_params'] = lparams

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs,sparams,description)

    # setup parameters "in time"
    t0 = 0
    dt = 0.1
    Tend = 2

    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    print('Init:',uinit.values)

    # call main function to get things done...
    uend,stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    # get stats and error
    extract_stats = grep_stats(stats,type='niter')
    sortedlist_stats = sort_stats(extract_stats,sortby='time')

    uex = P.u_exact(Tend)

    print('Error:',np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(uex.values,np.inf))

    print('Min/Max/Sum number of iterations: %s/%s/%s' %(min(entry[1] for entry in sortedlist_stats),
                                                         max(entry[1] for entry in sortedlist_stats),
                                                         sum(entry[1] for entry in sortedlist_stats)))
