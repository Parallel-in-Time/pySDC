
from pySDC import CollocationClasses as collclass

import numpy as np

from examples.heat1d.ProblemClass import heat1d
from examples.heat1d.TransferClass import mesh_to_mesh_1d

from examples.advection.ProblemClass import advection
from examples.advection.TransferClass import mesh_to_mesh_1d_periodic

from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order
import pySDC.PFASST_stepwise as mp
from pySDC import Log
from pySDC.Stats import grep_stats, sort_stats

import pySDC.Plugins.fault_tolerance as ft


if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    # logger = Log.setup_custom_logger('root')

    num_procs = 16

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-09

    sparams = {}
    sparams['maxiter'] = 50

    # # This comes as read-in for the problem class
    # pparams = {}
    # pparams['nu'] = 0.5
    # pparams['nvars'] = [255,127]
    #
    # # This comes as read-in for the transfer operations
    # tparams = {}
    # tparams['finter'] = True
    #
    # # Fill description dictionary for easy hierarchy creation
    # description = {}
    # description['problem_class'] = heat1d
    # description['problem_params'] = pparams
    # description['dtype_u'] = mesh
    # description['dtype_f'] = rhs_imex_mesh
    # description['collocation_class'] = collclass.CollGaussLobatto
    # description['num_nodes'] = 5
    # description['sweeper_class'] = imex_1st_order
    # description['level_params'] = lparams
    # description['transfer_class'] = mesh_to_mesh_1d
    # description['transfer_params'] = tparams
    #
    # # quickly generate block of steps
    # MS = mp.generate_steps(num_procs,sparams,description)
    #
    # # setup parameters "in time"
    # t0 = 0
    # dt = 0.5
    # Tend = 16*dt


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

    ft_iter = range(1,12)
    ft_step = range(0,16)
    ft_strategy = ['SPREAD']

    for strategy in ft_strategy:

        iter_count = np.zeros((len(ft_step),len(ft_iter)))

        xcnt = -1

        for step in ft_step:

            xcnt +=1

            ycnt = -1
            for iter in ft_iter:

                ycnt +=1

                ft.step = step
                ft.iter = iter
                ft.strategy = strategy

                # quickly generate block of steps
                MS = mp.generate_steps(num_procs,sparams,description)
                # get initial values on finest level
                P = MS[0].levels[0].prob
                uinit = P.u_exact(t0)

                # call main function to get things done...
                uend,stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

                # compute exact solution and compare
                uex = P.u_exact(Tend)

                print('error at time %s: %s' %(Tend,np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(
                    uex.values,np.inf)))

                extract_stats = grep_stats(stats,iter=-1,type='niter')
                sortedlist_stats = sort_stats(extract_stats,sortby='step')
                print(sortedlist_stats)
                niter = sortedlist_stats[-1][1]
                iter_count[xcnt,ycnt] = niter
                # print('PFASST needed %i iterations to converge' %niter)
                print('\n')

        print(iter_count)

        np.savez('advection_results_hf_'+strategy,iter_count=iter_count,description=description,ft_step=ft_step,
                 ft_iter=ft_iter)