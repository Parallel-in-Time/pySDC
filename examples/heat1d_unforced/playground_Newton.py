
from pySDC import CollocationClasses as collclass

import numpy as np

from examples.heat1d_unforced.ProblemClass import heat1d_unforced
from examples.heat1d_unforced.TransferClass import mesh_to_mesh_1d
from pySDC.datatype_classes.mesh import mesh
# from pySDC.datatype_classes.complex_mesh import mesh
from pySDC.sweeper_classes.generic_implicit import generic_implicit
import pySDC.PFASST_blockwise as mp
# import pySDC.PFASST_stepwise as mp
# import pySDC.Methods as mp
from pySDC import Log
# from pySDC.Stats import grep_stats, sort_stats

from pySDC.Plugins.sweeper_helper import get_Qd
from pySDC.sweeper_classes.linearized_implicit_fixed_parallel import linearized_implicit_fixed_parallel

from pySDC.Plugins.visualization_tools import show_residual_across_simulation



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
    pparams['nvars'] = [63]
    pparams['k'] = 2

    # This comes as read-in for the transfer operations (this is optional!)
    tparams = {}
    tparams['finter'] = False
    tparams['iorder'] = 2
    tparams['rorder'] = 1

    Nnodes = 5
    cclass = collclass.CollGaussRadau_Right

    # This comes as read-in for the sweeper class
    swparams = {}
    swparams['QI'] = 'Qpar'
    swparams_coarse = {}
    swparams_coarse['QI'] = 'LU'
    swparams['fixed_time_in_jacobian'] = 0
    swparams_coarse['fixed_time_in_jacobian'] = 0

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = heat1d_unforced
    description['problem_params'] = pparams
    description['dtype_u'] = mesh
    description['dtype_f'] = mesh
    description['collocation_class'] = cclass
    description['num_nodes'] = Nnodes
    # description['sweeper_class'] = generic_implicit
    description['sweeper_class'] = linearized_implicit_fixed_parallel
    description['sweeper_params'] = [swparams,swparams_coarse]
    # description['sweeper_params'] = [swparams]
    description['level_params'] = lparams
    description['transfer_class'] = mesh_to_mesh_1d
    description['transfer_params'] = tparams

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs,sparams,description)

    # setup parameters "in time"
    t0 = 0
    dt = 0.1
    Tend = num_procs*dt

    # get initial values on finest level
    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend,stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    # compute exact solution and compare
    uex = P.u_exact(Tend)

    print('error at time %s: %s' %(Tend,np.linalg.norm(uex.values-uend.values,np.inf)))


    # show_residual_across_simulation(stats,'res_vis_test.png')

    # extract_stats = grep_stats(stats,iter=-1,type='residual')
    # sortedlist_stats = sort_stats(extract_stats,sortby='step')
    # print(extract_stats,sortedlist_stats)