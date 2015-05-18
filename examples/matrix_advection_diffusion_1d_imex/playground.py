
from pySDC import CollocationClasses as collclass

import numpy as np

from examples.matrix_advection_diffusion_1d_imex.ProblemClass import advection_diffusion
from examples.matrix_advection_diffusion_1d_imex.TransferClass import mesh_to_mesh_1d_periodic
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order
import pySDC.PFASST_blockwise as mp
import pySDC.MatrixMethods as mmp
from pySDC import Log
from pySDC.Stats import grep_stats, sort_stats

if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 4

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-10

    sparams = {}
    sparams['maxiter'] = 35

    # This comes as read-in for the problem class
    pparams = {}
    pparams['c'] = 1.0
    pparams['nu'] = 0.1
    pparams['nvars'] = [64, 32]
    pparams['order'] = [6]

    # This comes as read-in for the all kind of generating options for the matrix classes
    mparams = {}
    mparams['sparse_format'] = "array"

    # This comes as read-in for the transfer operations
    tparams = {}
    tparams['finter'] = True
    tparams['sparse_format'] = "array"
    tparams['interpolation_order'] = [[3, 10]]*num_procs
    tparams['restriction_order'] = [[3, 10]]*num_procs
    tparams['interpolation_order'] = [[3, 10]]*num_procs
    tparams['restriction_order'] = [[3, 10]]*num_procs
    tparams['t_interpolation_order'] = [2]*num_procs
    tparams['t_restriction_order'] = [2]*num_procs


    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = advection_diffusion
    description['problem_params'] = pparams
    description['dtype_u'] = mesh
    description['dtype_f'] = rhs_imex_mesh
    description['collocation_class'] = collclass.CollGaussLobatto
    description['num_nodes'] = 5
    description['sweeper_class'] = imex_1st_order
    description['level_params'] = lparams
    description['transfer_class'] = mesh_to_mesh_1d_periodic
    description['transfer_params'] = tparams

    # Options for run_linear_pfasst
    linpparams = {}
    linpparams['run_type'] = "tolerance"
    linpparams['norm'] = lambda x: np.linalg.norm(x, np.inf)
    linpparams['tol'] = lparams['restol']*100
    linpparams['tol'] = lparams['restol']*100

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs,sparams,description)

    # setup parameters "in time"
    t0 = 0.0
    dt = 0.005
    Tend = 5*dt
    print "cfl:", pparams['nu']*(pparams['nvars'][0]**2)*dt
    # get initial values on finest level
    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend,stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    # start with the analysis using the iteration matrix of PFASST

    transfer_list = mmp.generate_transfer_list(MS, description['transfer_class'], **tparams)
    lin_pfasst = mmp.generate_LinearPFASST(MS, transfer_list, uinit.values, **tparams)
    u_0 = np.kron(np.asarray([1]*description['num_nodes']+[1]*description['num_nodes']*(num_procs-1)),
                  uinit.values)
    res, u = mmp.run_linear_pfasst(lin_pfasst, u_0, linpparams)
    all_nodes = mmp.get_all_nodes(MS, t0)
    print "Residuals:\n", res, "\nNumber of iterations: ", len(res)-1
    u_end_split = np.split(u[-1], num_procs*description['num_nodes'])
    print "Spectral Radius:\t", lin_pfasst.spectral_radius()
    lfa = mmp.LFAForLinearPFASST(lin_pfasst, MS, transfer_list, debug=True)
    print "lfa:"
    print lfa.asymptotic_conv_factor()
    res, u = mmp.run_linear_pfasst(lin_pfasst, u_0, linpparams)

    # compute exact solution and compare
    uex = P.u_exact(Tend)

    # print('error at time %s: %s' %(Tend,np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(
    #     uex.values,np.inf)))

    print(' absolute difference between pfasst and lin_pfasst at time %s: %s' %(Tend,np.linalg.norm(u_end_split[-1]-uend.values, np.inf)))
    print(u_end_split[-1])
    print(uend.values)
    print(uend.values - u_end_split[-1])
    # extract_stats = grep_stats(stats,iter=-1,type='residual')
    # sortedlist_stats = sort_stats(extract_stats,sortby='step')
    # print(extract_stats,sortedlist_stats)
