
from pySDC import CollocationClasses as collclass

import numpy as np

from examples.heat1d.ProblemClass import heat1d
from examples.heat1d.TransferClass import mesh_to_mesh_1d

from examples.advection_1d_implicit.ProblemClass import advection
from examples.advection_1d_implicit.TransferClass import mesh_to_mesh_1d_periodic

from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order
import pySDC.PFASST_stepwise as mp
import pySDC.Stats as st
# from pySDC.Stats import grep_stats, sort_stats

from matplotlib import rc
import matplotlib.pyplot as plt

import pySDC.Plugins.fault_tolerance as ft
from pySDC.Plugins.visualization_tools import show_residual_across_simulation


if __name__ == "__main__":
    """
        This routine generates the heatmaps showing the residual for a node failures at step n and iteration k
    """

    # set global logger (remove this if you do not want the output at all)
    # logger = Log.setup_custom_logger('root')

    num_procs = 16

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-09

    sparams = {}
    sparams['maxiter'] = 50

    # choose the strategy, the setup, the iteration/step where the fault should happen
    ft_strategy = ['NOFAULT','SPREAD','SPREAD_PREDICT','INTERP','INTERP_PREDICT']
    ft_setup = 'ADVECTION'
    # ft_setup = 'HEAT'
    ft_step = 7
    ft_iter = 7

    if ft_setup is 'HEAT':

        # This comes as read-in for the problem class
        pparams = {}
        pparams['nu'] = 0.5
        pparams['nvars'] = [255,127]

        # This comes as read-in for the transfer operations
        tparams = {}
        tparams['finter'] = True
        tparams['iorder'] = 6
        tparams['rorder'] = 2

        # Fill description dictionary for easy hierarchy creation
        description = {}
        description['problem_class'] = heat1d
        description['problem_params'] = pparams
        description['dtype_u'] = mesh
        description['dtype_f'] = rhs_imex_mesh
        description['collocation_class'] = collclass.CollGaussLobatto
        description['num_nodes'] = 5
        description['sweeper_class'] = imex_1st_order
        description['level_params'] = lparams
        description['transfer_class'] = mesh_to_mesh_1d
        description['transfer_params'] = tparams

        # quickly generate block of steps
        MS = mp.generate_steps(num_procs,sparams,description)

        # setup parameters "in time"
        t0 = 0
        dt = 0.5
        Tend = 16*dt

    elif ft_setup is 'ADVECTION':

        # This comes as read-in for the problem class
        pparams = {}
        pparams['c'] = 1.0
        pparams['nvars'] = [256,128]
        pparams['order'] = [2,2]

        # This comes as read-in for the transfer operations
        tparams = {}
        tparams['finter'] = True

        # Fill description dictionary for easy hierarchy creation
        description = {}
        description['problem_class'] = advection
        description['problem_params'] = pparams
        description['dtype_u'] = mesh
        description['dtype_f'] = rhs_imex_mesh
        description['collocation_class'] = collclass.CollGaussLobatto
        description['num_nodes'] = 5
        description['sweeper_class'] = imex_1st_order
        description['level_params'] = lparams
        description['transfer_class'] = mesh_to_mesh_1d_periodic
        description['transfer_params'] = tparams

        # setup parameters "in time"
        t0 = 0.0
        dt = 0.125
        Tend = 16*dt

    else:

        print('setup not implemented, aborting...',setup)
        exit()


    # loop over all stategies and check how things evolve
    for strategy in ft_strategy:

        ft.strategy = strategy
        ft.hard_step = ft_step
        ft.hard_iter = ft_iter

        # quickly generate block of steps
        MS = mp.generate_steps(num_procs,sparams,description)
        # get initial values on finest level
        P = MS[0].levels[0].prob
        uinit = P.u_exact(t0)

        # call main function to get things done...
        uend,stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

        # stats magic: get iteration counts to find maxiter/niter
        extract_stats = st.grep_stats(stats,iter=-1,type='niter')
        sortedlist_stats = st.sort_stats(extract_stats,sortby='step')
        niter = sortedlist_stats[-1][1]
        print('Iterations:',niter)

        residual = np.zeros((niter,num_procs))
        residual[:] = -99

        #stats magic: extract all residuals (steps vs. iterations)
        extract_stats = st.grep_stats(stats,level=-1,type='residual')
        for k,v in extract_stats.items():
            step = getattr(k,'step')
            iter = getattr(k,'iter')
            if iter is not -1:
                residual[iter-1,step] = np.log10(v)
        print('')
        np.savez(ft_setup+'_steps_vs_iteration_hf_'+strategy,residual=residual,ft_step=ft.hard_step,ft_iter=ft.hard_iter)
