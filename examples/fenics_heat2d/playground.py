
from pySDC import CollocationClasses as collclass

import numpy as np

from ProblemClass import fenics_heat2d
from fenics_mesh import fenics_mesh, rhs_fenics_mesh
from pySDC.sweeper_classes.mass_matrix_imex import mass_matrix_imex
import pySDC.Methods as mp
from pySDC import Log
from pySDC.Stats import grep_stats, sort_stats

if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 1

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 3E-12

    sparams = {}
    sparams['maxiter'] = 10

    # This comes as read-in for the problem class
    pparams = {}
    pparams['alpha'] = 3
    pparams['beta'] = 1.2
    pparams['nvars'] = [[2,2]]
    pparams['t0'] = 0.0 # ugly, but necessary to set up ProblemClass

    # This comes as read-in for the transfer operations
    # tparams = {}
    # tparams['finter'] = True

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = fenics_heat2d
    description['problem_params'] = pparams
    description['dtype_u'] = fenics_mesh
    description['dtype_f'] = rhs_fenics_mesh
    description['collocation_class'] = collclass.CollGaussLobatto
    description['num_nodes'] = 3
    description['sweeper_class'] = mass_matrix_imex
    description['level_params'] = lparams
    # description['transfer_class'] = mesh_to_mesh_1d
    # description['transfer_params'] = tparams

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs,sparams,description)

    exit()

    # setup parameters "in time"
    t0 = MS[0].levels[0].prob.t0
    dt = 0.3
    Tend = 1*dt

    # get initial values on finest level
    P = MS[0].levels[0].prob

    uinit = P.u_init(t0)

    # call main function to get things done...
    uend,stats = mp.run_pfasst_serial(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    # compute exact solution and compare
    # uex = P.u_exact(Tend)

    # print('error at time %s: %s' %(Tend,np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(
    #     uex.values,np.inf)))

    extract_stats = grep_stats(stats,iter=-1,type='residual')
    sortedlist_stats = sort_stats(extract_stats,sortby='step')
    print(extract_stats,sortedlist_stats)