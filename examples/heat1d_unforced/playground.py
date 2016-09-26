import numpy as np
import pySDC.deprecated.PFASST_blockwise_old as mp

from examples.heat1d_unforced.HookClass import error_output
from examples.heat1d_unforced.ProblemClass import heat1d_unforced
from examples.heat1d_unforced.TransferClass import mesh_to_mesh_1d
from implementations.datatype_classes import mesh
from implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC import CollocationClasses as collclass

# import pySDC.PFASST_stepwise as mp
# import pySDC.Methods as mp
# from pySDC.Stats import grep_stats, sort_stats



if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    # logger = Log.setup_custom_logger('root')

    num_procs = 1

    # This comes as read-in for the level class  (this is optional!)
    lparams = {}
    lparams['restol'] = 1E-10

    # This comes as read-in for the step class (this is optional!)
    sparams = {}
    sparams['maxiter'] = 20

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nu'] = 1
    pparams['nvars'] = [31]
    pparams['k'] = 1

    # This comes as read-in for the transfer operations (this is optional!)
    tparams = {}
    tparams['finter'] = False
    tparams['iorder'] = 2
    tparams['rorder'] = 1

    # This comes as read-in for the sweeper class
    swparams = {}
    swparams['QI'] = 'LU'
    swparams['collocation_class'] = collclass.CollGaussRadau_Right
    swparams['num_nodes'] = 5

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = heat1d_unforced
    description['problem_params'] = pparams
    description['dtype_u'] = mesh
    description['dtype_f'] = mesh
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = swparams
    description['level_params'] = lparams
    description['transfer_class'] = mesh_to_mesh_1d
    description['transfer_params'] = tparams
    description['hook_class'] = error_output

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs,sparams,description)

    # setup parameters "in time"
    t0 = 0
    dt = 0.5
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