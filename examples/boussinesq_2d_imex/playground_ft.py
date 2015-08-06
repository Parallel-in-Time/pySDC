
from pySDC import CollocationClasses as collclass

import numpy as np

from ProblemClass import boussinesq_2d_imex
from examples.boussinesq_2d_imex.TransferClass import mesh_to_mesh_2d
from examples.boussinesq_2d_imex.HookClass import plot_solution

from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order
import pySDC.PFASST_stepwise as mp
from pySDC import Log
from pySDC.Stats import grep_stats, sort_stats

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from unflatten import unflatten

import pySDC.Plugins.fault_tolerance as ft


if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 1

    ft_strategy = ['NOFAULT']#,'SPREAD','SPREAD_PREDICT','INTERP','INTERP_PREDICT']
    ft_step = 7
    ft_iter = 7

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-12
    
    swparams = {}
    swparams['collocation_class'] = collclass.CollGaussLobatto
    swparams['num_nodes'] = 3
    swparams['do_LU'] = True

    sparams = {}
    sparams['maxiter'] = 2

    # setup parameters "in time"
    t0     = 0
    Tend   = 3
    Nsteps =  1
    dt = Tend/float(Nsteps)

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nvars']    = [(4,300,10)]
    pparams['u_adv']    = 0.02
    pparams['c_s']      = 0.3
    pparams['Nfreq']    = 0.01
    pparams['x_bounds'] = [(-150.0, 150.0)]
    pparams['z_bounds'] = [(   0.0,  10.0)]
    pparams['order']    = [0, 0] # [fine_level, coarse_level]

    # This comes as read-in for the transfer operations
    tparams = {}
    tparams['finter'] = False

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class']     = boussinesq_2d_imex
    description['problem_params']    = pparams
    description['dtype_u']           = mesh
    description['dtype_f']           = rhs_imex_mesh
    description['sweeper_params']    = swparams
    description['sweeper_class']     = imex_1st_order
    description['level_params']      = lparams
    # description['hook_class']        = plot_solution
    description['transfer_class']    = mesh_to_mesh_2d
    description['transfer_params']   = tparams

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

        P.report_log()

    # extract_stats = grep_stats(stats,iter=-1,type='residual')
    # sortedlist_stats = sort_stats(extract_stats,sortby='step')
    # print(extract_stats,sortedlist_stats)