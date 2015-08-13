
from pySDC import CollocationClasses as collclass

import numpy as np

from examples.heat1d.ProblemClass import heat1d
from examples.heat1d.TransferClass import mesh_to_mesh_1d

from examples.advection_1d_implicit.ProblemClass import advection
from examples.advection_1d_implicit.TransferClass import mesh_to_mesh_1d_periodic

from examples.penningtrap.ProblemClass import penningtrap,penningtrap_coarse
from examples.penningtrap.TransferClass import particles_to_particles
from pySDC.datatype_classes.particles import particles, fields
from pySDC.sweeper_classes.boris_2nd_order import boris_2nd_order

from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order

import pySDC.PFASST_stepwise as mp
from pySDC import Log
from pySDC.Stats import grep_stats, sort_stats

import pySDC.Plugins.fault_tolerance as ft


if __name__ == "__main__":
    """
        This routine generates the heatmaps showing the iteration counts for node failures at different
        steps and iterations
    """

    # set global logger (remove this if you do not want the output at all)
    # logger = Log.setup_custom_logger('root')

    num_procs = 16

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-09

    sparams = {}
    sparams['maxiter'] = 50

    # choose the strategy and the setup
    ft_strategy = ['SPREAD','SPREAD_PREDICT','INTERP','INTERP_PREDICT']
    # ft_setup = 'HEAT'
    ft_setup = 'ADVECTION'

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
        tparams['iorder'] = 6
        tparams['rorder'] = 2

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

        print('setup not implemented, aborting...',ft_setup)
        exit()

    # do a reference run without any faults to see how things would look like (and to get maxiter/ref_niter)
    ft.strategy = 'NOFAULT'

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs,sparams,description)
    # get initial values on finest level
    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend,stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    # stats magic: get niter
    extract_stats = grep_stats(stats,iter=-1,type='niter')
    sortedlist_stats = sort_stats(extract_stats,sortby='step')
    ref_niter = sortedlist_stats[-1][1]

    print('Will sweep over %i steps and %i iterations now...' %(num_procs,ref_niter))

    # loop over all strategies
    for strategy in ft_strategy:

        ft_iter = range(1,ref_niter+1)
        ft_step = range(0,num_procs)

        print('------------------------------------------ working on strategy ',strategy)

        iter_count = np.zeros((len(ft_step),len(ft_iter)))

        # loop over all steps
        xcnt = -1
        for step in ft_step:

            xcnt +=1

            # loop over all iterations
            ycnt = -1
            for iter in ft_iter:

                ycnt +=1

                ft.hard_step = step
                ft.hard_iter = iter
                ft.strategy = strategy

                # quickly generate block of steps
                MS = mp.generate_steps(num_procs,sparams,description)
                # get initial values on finest level
                P = MS[0].levels[0].prob
                uinit = P.u_exact(t0)

                # call main function to get things done...
                uend,stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

                # stats magic: get niter
                extract_stats = grep_stats(stats,iter=-1,type='niter')
                sortedlist_stats = sort_stats(extract_stats,sortby='step')
                print(sortedlist_stats)
                niter = sortedlist_stats[-1][1]
                iter_count[xcnt,ycnt] = niter

        print(iter_count)

        np.savez(ft_setup+'_results_hf_'+strategy,iter_count=iter_count,description=description,ft_step=ft_step,
                 ft_iter=ft_iter)