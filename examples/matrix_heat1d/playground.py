
from pySDC import CollocationClasses as collclass

import numpy as np
import scipy.linalg as la

from examples.matrix_heat1d.ProblemClass import heat1d
from examples.matrix_heat1d.TransferClass import mesh_to_mesh_1d

from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order
import pySDC.PFASST_blockwise as mp
from pySDC import Log
from pySDC.Stats import grep_stats, sort_stats
import pySDC.MatrixMethods as mmp

from pySDC.tools.transfer_tools import interpolate_to_t_end
if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 4

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 3E-9

    sparams = {}
    sparams['maxiter'] = 50

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nu'] = 0.1
    pparams['nvars'] = [63,31]
    # pparams['nvars'] = [15, 7]

    # This comes as read-in for the all kind of generating options for the matrix classes
    mparams = {}
    mparams['sparse_format'] = "array"

    # This comes as read-in for the transfer operations
    tparams = {}
    tparams['finter'] = False
    tparams['sparse_format'] = "array"
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
    description['collocation_class'] = collclass.CollGaussLobatto
    description['num_nodes'] = 3
    description['sweeper_class'] = imex_1st_order
    description['level_params'] = lparams
    description['transfer_class'] = mesh_to_mesh_1d
    description['transfer_params'] = tparams

    # Options for run_linear_pfasst
    linpparams = {}
    linpparams['run_type'] = "tolerance"
    linpparams['norm'] = lambda x: np.linalg.norm(x, np.inf)
    linpparams['tol'] = 3E-9

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs,sparams,description)


    # setup parameters "in time"
    t0 = 0
    dt = 0.125
    Tend = 4*dt
    print "cfl:", pparams['nu']*(pparams['nvars'][0]**2)*dt
    # get initial values on finest level
    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    # print uinit
    # call main function to get things done...
    uend, stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    # MS = mp.generate_steps(num_procs,sparams,description)
    # u_0 = []
    # for S,p in zip(MS,range(len(MS))):
    #     # call predictor from sweeper
    #     S.status.dt = dt # could have different dt per step here
    #     S.status.time = t0 + sum(MS[j].status.dt for j in range(p))
    #     S.init_step(uinit)
    #     S.levels[0].sweep.predict()
    #
    # MS = mp.predictor(MS)
    #
    # for S in MS:
    #     for u in S.u[1:]:
    #         u_0.append(u)


    # start with the analysis using the iteration matrix of PFASST

    transfer_list = mmp.generate_transfer_list(MS, description['transfer_class'], **tparams)
    lin_pfasst = mmp.generate_LinearPFASST(MS, transfer_list, uinit.values, **tparams)
    # print lin_pfasst.spectral_radius()
    # print
    # lin_pfasst.check_condition_numbers(p=2)
    u_0 = np.kron(np.asarray([1]*description['num_nodes']+[1]*description['num_nodes']*(num_procs-1)),
                  uinit.values)

    res, u = mmp.run_linear_pfasst(lin_pfasst, u_0, linpparams)
    all_nodes = mmp.get_all_nodes(MS, t0)
    print "Residuals:\n", res, "\nNumber of iterations: ", len(res)-1
    u_end_split = np.split(u[-1], num_procs*description['num_nodes'])


    uex = P.u_exact(Tend)
    print "relative error per linpfasst iteration"
    for u in u[1:]:
        last_u = np.split(u, num_procs*description['num_nodes'])[-1]
        print np.linalg.norm(uex.values-last_u, np.inf)/np.linalg.norm(uex.values, np.inf)

    print('matrix error at time %s: %s' %(Tend, np.linalg.norm(uex.values-u_end_split[-1], np.inf)/np.linalg.norm(
        uex.values, 2)))
    print('non matrix error at time %s: %s' %(Tend,np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(
        uex.values, 2)))
    print('difference between pfasst and lin_pfasst at time %s: %s' %(Tend,np.linalg.norm(u_end_split[-1]-uend.values, np.inf)/np.linalg.norm(
        uex.values, 2)))
    # extract_stats = grep_stats(stats, type='residual')
    # sortedlist_stats = sort_stats(extract_stats, sortby='step')
    # for item in sortedlist_stats:
    #     print(item)
    # print(extract_stats, sortedlist_stats)
