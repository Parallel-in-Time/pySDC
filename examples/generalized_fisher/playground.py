import matplotlib.pyplot as plt
import numpy as np
import pySDC.deprecated.PFASST_blockwise_old as mp

from examples.generalized_fisher.ProblemClass import generalized_fisher
from implementations.datatype_classes import mesh
from implementations.sweeper_classes.linearized_implicit_fixed_parallel import linearized_implicit_fixed_parallel
from pySDC import CollocationClasses as collclass
from pySDC import Log
from pySDC.plugins.sweeper_helper import get_Qd

if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 1

    # This comes as read-in for the level class  (this is optional!)
    lparams = {}
    lparams['restol'] = 1E-10

    # This comes as read-in for the step class (this is optional!)
    sparams = {}
    sparams['maxiter'] = 20
    sparams['fine_comm'] = True

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nu'] = 1
    pparams['nvars'] = [64]
    pparams['lambda0'] = 1.0
    pparams['maxiter'] = 50
    pparams['newton_tol'] = 1E-12

    Nnodes = 5
    cclass = collclass.CollGaussRadau_Right

    # This comes as read-in for the sweeper class
    swparams = {}
    swparams['QI'] = get_Qd(cclass,Nnodes=Nnodes,qd_type='LU')
    swparams['fixed_time_in_jacobian'] = 0

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = generalized_fisher
    description['problem_params'] = pparams
    description['dtype_u'] = mesh
    description['dtype_f'] = mesh
    description['collocation_class'] = cclass
    description['num_nodes'] = Nnodes
    # description['sweeper_class'] = generic_implicit
    description['sweeper_class'] = linearized_implicit_fixed_parallel
    description['sweeper_params'] = [swparams]
    description['level_params'] = lparams

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs,sparams,description)

    # setup parameters "in time"
    t0 = 0
    dt = 0.1
    Tend = dt

    # get initial values on finest level
    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    plt.plot(uinit.values,'r--',lw=2)

    # call main function to get things done...
    uend,stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    # compute exact solution and compare
    uex = P.u_exact(Tend)

    print('error at time %s: %s' %(Tend,np.linalg.norm(uex.values-uend.values,np.inf)))

    plt.plot(uend.values,'bs',lw=2)
    plt.plot(uex.values,'gd',lw=2)
    # plt.show()
    # show_residual_across_simulation(stats,'res_vis_test.png')

    # extract_stats = grep_stats(stats,iter=-1,type='residual')
    # sortedlist_stats = sort_stats(extract_stats,sortby='step')
    # print(extract_stats,sortedlist_stats)