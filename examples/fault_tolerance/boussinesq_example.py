
from pySDC import CollocationClasses as collclass

from examples.boussinesq_2d_imex.ProblemClass import boussinesq_2d_imex
from examples.boussinesq_2d_imex.TransferClass import mesh_to_mesh_2d
from examples.boussinesq_2d_imex.HookClass import plot_solution

from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order
import pySDC.PFASST_stepwise as mp

from pySDC import Log
from pySDC.Stats import grep_stats, sort_stats

import numpy as np

import pySDC.Plugins.fault_tolerance as ft


if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 16

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-06

    swparams = {}
    swparams['collocation_class'] = collclass.CollGaussLobatto
    swparams['num_nodes'] = 3
    swparams['do_LU'] = True

    sparams = {}
    sparams['maxiter'] = 50

    # setup parameters "in time"
    t0     = 0
    Tend   = 960
    Nsteps =  320
    dt = Tend/float(Nsteps)

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nvars']    = [(4,450,30), (4,450,30)]
    pparams['u_adv']    = 0.02
    pparams['c_s']      = 0.3
    pparams['Nfreq']    = 0.01
    pparams['x_bounds'] = [(-150.0, 150.0)]
    pparams['z_bounds'] = [(   0.0,  10.0)]
    pparams['order']    = [4, 2] # [fine_level, coarse_level]
    pparams['order_upw'] = [5, 1]
    pparams['gmres_maxiter'] = [50, 50]
    pparams['gmres_restart'] = [10, 10]
    pparams['gmres_tol']     = [1e-10, 1e-10]

    # This comes as read-in for the transfer operations
    tparams = {}
    tparams['finter'] = True

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class']     = boussinesq_2d_imex
    description['problem_params']    = pparams
    description['dtype_u']           = mesh
    description['dtype_f']           = rhs_imex_mesh
    description['sweeper_params']    = swparams
    description['sweeper_class']     = imex_1st_order
    description['level_params']      = lparams
    description['hook_class']        = plot_solution
    description['transfer_class']    = mesh_to_mesh_2d
    description['transfer_params']   = tparams

    ft.hard_random = 0.03

    strategies = ['NOFAULT']
    # strategies = ['NOFAULT','SPREAD','INTERP','INTERP_PREDICT','SPREAD_PREDICT']

    for strategy in strategies:

        print('------------------------------------------ working on strategy ',strategy)
        ft.strategy = strategy

        # quickly generate block of steps
        MS = mp.generate_steps(num_procs,sparams,description)

        # get initial values on finest level
        P = MS[0].levels[0].prob
        uinit = P.u_exact(t0)

        cfl_advection = pparams['u_adv']*dt/P.h[0]
        cfl_acoustic_hor = pparams['c_s']*dt/P.h[0]
        cfl_acoustic_ver = pparams['c_s']*dt/P.h[1]
        print ("CFL number of advection: %4.2f" % cfl_advection)
        print ("CFL number of acoustics (horizontal): %4.2f" % cfl_acoustic_hor)
        print ("CFL number of acoustics (vertical):   %4.2f" % cfl_acoustic_ver)

        # call main function to get things done...
        uend,stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

        P.report_log()

        # stats magic: get all residuals
        extract_stats = grep_stats(stats,level=-1,type='residual')

        # find boundaries
        maxsteps = 0
        maxiter = 0
        minres = 0
        maxres = -99
        for k,v in extract_stats.items():
            maxsteps = max(maxsteps,getattr(k,'step'))
            maxiter = max(maxiter,getattr(k,'iter'))
            minres = min(minres,np.log10(v))
            maxres = max(maxres,np.log10(v))

        # fill residual array
        residual = np.zeros((maxiter,maxsteps+1))
        residual[:] = 0
        for k,v in extract_stats.items():
            step = getattr(k,'step')
            iter = getattr(k,'iter')
            if iter is not -1:
                residual[iter-1,step] = v

        # stats magic: get niter (probably redundant with maxiter)
        extract_stats = grep_stats(stats,iter=-1,type='niter')
        iter_count = np.zeros(maxsteps+1)
        for k,v in extract_stats.items():
            step = getattr(k,'step')
            iter_count[step] = v
        print(iter_count)

        # np.savez('SDC_GRAYSCOTT_stats_hf_'+ft.strategy+'_new',residual=residual,iter_count=iter_count,hard_stats=ft.hard_stats)
        np.savez('PFASST_BOUSSINESQ_stats_hf_'+ft.strategy+'_P'+str(num_procs),residual=residual,iter_count=iter_count,hard_stats=ft.hard_stats)

