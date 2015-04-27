
from pySDC import CollocationClasses as collclass

import numpy as np
import scipy.linalg as la

from examples.matrix_heat1d.ProblemClass import heat1d
from examples.matrix_heat1d.TransferClass import mesh_to_mesh_1d

from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order
import pySDC.Methods as mp
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
    lparams['restol'] = 3E-12

    sparams = {}
    sparams['maxiter'] = 10

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nu'] = 0.1
    pparams['nvars'] = [15, 7]

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
    # what to do with lin pfasst
    # one could compute the spectral radius
    # print("Spektral radius of PFASST:", lin_pfasst.spectral_radius())
    # Or one could compute some steps and see how the error converges
    # first we need an initial value
    u_0 = lin_pfasst.c
    M = lin_pfasst.M
    first_block = M[:45, :45]
    first_block_inv = la.inv(first_block)
    u_sol = la.inv(M).dot(u_0)
    u_sol_split = np.split(u_sol, 12)
    u_sol_fblock = first_block_inv.dot(np.kron(np.ones(3), uinit.values))
    # print u_sol_split[0],"\n",u_sol_fblock
    all_nodes = mmp.get_all_nodes(MS)
    u_sol_at_tend = interpolate_to_t_end(all_nodes[:3]/dt, u_sol_split[-3:])

    # for p, i in zip(np.split(u_0, 12), all_nodes):
    #     print i, "\n", p, "\n"


    u = [u_0]
    res = [u_0 - M.dot(u_0)]
    numb_sweeps = 10
    for i in range(numb_sweeps):
        u.append(lin_pfasst.step(u[-1]))
        res.append(u_0-M.dot(u[-1]))
    print "Residuum LinearPFASST:\n", res[-1]
    # compute exact solution and compare
    uex = P.u_exact(Tend)
    for i in u:
        print uex.values - np.split(i, 12)[-1]
    uex_last_node = P.u_exact(all_nodes[-1])
    # print uex.values,"\n",u_sol_at_tend
    # print uex_last_node.values,"\n",u_sol_split[-1]
    # testen mal was der erste block so produziert
    # ha = 3
    # print all_nodes
    # print P.u_exact(all_nodes[ha]).values, "\n", u_sol_split[ha]
    # print P.u_exact(all_nodes[0]).values - u_sol_fblock[:15]

    # get the first iterative solver in the list of linpfasst and test ist
    # basically test if the sdc part is computed right
    first_it_solv =lin_pfasst.multi_step_solver.it_list[0]
    # print first_it_solv is lin_pfasst.block_diag_solver.it_list[0]
    # print first_it_solv.M
    # check with Q_mat and System mat
    print MS[0].levels[0].sweep.coll.Qmat[1:,1:]#.dot(np.ones(3))
    # print MS[0].levels[0].sweep.coll.nodes
    # print MS[0].levels[0].prob.system_matrix.toarray()[0,0]
    # print 1-0.125*MS[0].levels[0].prob.system_matrix.toarray()[0,0]*MS[0].levels[0].sweep.coll.Qmat[1,1]
    # das klappt ja schonmal
    # testen wi

    u_sdc = [np.kron(np.ones(3), uinit.values).flatten()]
    print np.kron(np.ones(3), uinit.values).flatten().shape

    res_sdc = [u_sdc[0] - first_it_solv.M]

    for i in range(numb_sweeps):
        u_sdc.append(first_it_solv.step(u_sdc[-1]))
        res_sdc.append(u_sdc[0]-first_it_solv.M.dot(u_sdc[-1]))
        # print res_sdc[-1]

    u_sol_fblock = la.inv(first_it_solv.M).dot(u_sdc[0])
    print P.u_exact(all_nodes[2]).values - u_sol_fblock[-15:]
    print('error at time %s: %s' %(Tend,np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(
        uex.values, np.inf)))

    extract_stats = grep_stats(stats, iter=-1, type='residual')
    sortedlist_stats = sort_stats(extract_stats, sortby='step')
    print(extract_stats, sortedlist_stats)
