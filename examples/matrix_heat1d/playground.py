
from pySDC import CollocationClasses as collclass

import numpy as np

from examples.matrix_heat1d.ProblemClass import heat1d
from examples.matrix_heat1d.TransferClass import mesh_to_mesh_1d

from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order
import pySDC.Methods as mp
from pySDC import Log
from pySDC.Stats import grep_stats, sort_stats
import pySDC.MatrixMethods as mmp
if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 4

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 3E-12

    sparams = {}
    sparams['maxiter'] = 10

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nu'] = 0.1
    pparams['nvars'] = [127,63]

    # This comes as read-in for the all kind of generating options for the matrix classes
    mparams = {}
    mparams['sparse_format'] = "dense"

    # This comes as read-in for the transfer operations
    tparams = {}
    tparams['finter'] = True
    tparams['sparse_format'] = "dense"
    tparams['interpolation_order'] = [[3, 10]]*num_procs
    tparams['restriction_order'] = [[3, 10]]*num_procs
    tparams['interpolation_order'] = [[3, 10]]*num_procs
    tparams['restriction_order'] = [[3, 10]]*num_procs
    tparams['t_interpolation_order'] = [2]*num_procs
    tparams['t_restriction_order'] = [2]*num_procs


    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = heat1d
    description['problem_params'] = pparams
    description['dtype_u'] = mesh
    description['dtype_f'] = rhs_imex_mesh
    description['collocation_class'] = collclass.CollGaussLegendre
    description['num_nodes'] = 3
    description['sweeper_class'] = imex_1st_order
    description['level_params'] = lparams
    description['transfer_class'] = mesh_to_mesh_1d
    description['transfer_params'] = tparams

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs,sparams,description)

    # setup parameters "in time"
    t0 = 0
    dt = 0.125
    Tend = 4*dt

    # get initial values on finest level
    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    # print uinit
    # call main function to get things done...
    uend,stats = mp.run_pfasst_serial(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    # start with the analysis using the iteration matrix of PFASST

    transfer_list = mmp.generate_transfer_list(MS, description['transfer_class'], **tparams)
    lin_pfasst = mmp.generate_LinearPFASST(MS, transfer_list, uinit.values, **tparams)

    # compute exact solution and compare
    uex = P.u_exact(Tend)

    print('error at time %s: %s' %(Tend,np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(
        uex.values, np.inf)))

    extract_stats = grep_stats(stats, iter=-1, type='residual')
    sortedlist_stats = sort_stats(extract_stats, sortby='step')
    print(extract_stats, sortedlist_stats)
