
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



def run_with(opts, debug=False):
    """
    :param options:
    :return: (n_it_pfasst, n_it_lin_pfasst, spec_rad, lfa_spec_rad)
    """
    # set global logger (remove this if you do not want the output at all)
    # logger = Log.setup_custom_logger('root')
    #
    # make empty class
    opt = mmp.Bunch()
    for k, v in opts.items():
        setattr(opt, k, v)
    t0 = opt.t0
    dt = opt.dt
    Tend = opt.num_procs * dt
    cfl = opt.pparams['nu']*(opt.pparams['nvars'][0]**2)*dt

        # quickly generate block of steps
    MS = mp.generate_steps(opt.num_procs, opt.sparams, opt.description)
    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    uend, stats = mp.run_pfasst(MS, u0=uinit, t0=t0, dt=dt, Tend=Tend)

    # get max_iter from pfasst and min residual
    max_iter = 0.0
    min_res = 1.0
    for tup in sort_stats(stats, 'type'):
        if tup[0] is 'niter' and tup[1] > max_iter:
            max_iter = tup[1]
        elif tup[0] is 'residual' and tup[1] < min_res:
            min_res = tup[1]

    # initialize linear_pfasst
    transfer_list = mmp.generate_transfer_list(MS, opt.description['transfer_class'], **opt.tparams)
    lin_pfasst = mmp.generate_LinearPFASST(MS, transfer_list, uinit.values, **opt.tparams)
    lfa = mmp.LFAForLinearPFASST(lin_pfasst, MS, transfer_list, debug=True)
    spec_rad = max(lin_pfasst.spectral_radius(ka=8, tolerance=1e-7))
    lfa_asymp_conv = lfa.asymptotic_conv_factor()

    # run linear_pfasst to get the iteration number of lin_pfasst

    # first the initial value for linear_pfasst
    u_0 = np.kron(np.asarray([1]*opt.description['num_nodes']+[1]*opt.description['num_nodes']*(opt.num_procs-1)),
                  uinit.values)
    res, u = mmp.run_linear_pfasst(lin_pfasst, u_0, opt.linpparams)

    # get the number of iteration which where necesary to achieve a certain error
    uex = P.u_exact(Tend)
    it_from_error = 0
    last_u = map(lambda x: np.split(x, opt.num_procs*opt.description['num_nodes'])[-1],u)
    lin_pfasst_err = map(lambda x: np.linalg.norm(uex.values-x, np.inf)/np.linalg.norm(uex.values, np.inf), last_u)
    min_err_lin_pfasst = np.min(lin_pfasst_err)
    it_from_error = np.sum(np.abs((lin_pfasst_err - min_err_lin_pfasst)) > 2*min_err_lin_pfasst)
    # print it_from_error
    min_err = np.linalg.norm(uex.values-uend.values, np.inf)/np.linalg.norm(uex.values, np.inf)

    if debug:
        u_end_split = np.split(u[-1], opt.num_procs*opt.description['num_nodes'])
        print "cfl:", cfl
        print "relative error per linpfasst iteration"
        for u in u[1:]:
            last_u = np.split(u, opt.num_procs*opt.description['num_nodes'])[-1]
            print np.linalg.norm(uex.values-last_u, np.inf)/np.linalg.norm(uex.values, np.inf)

        print('matrix error at time %s: %s' %(Tend, np.linalg.norm(uex.values-u_end_split[-1], np.inf)/np.linalg.norm(
            uex.values, 2)))
        print('non matrix error at time %s: %s' %(Tend,np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(
            uex.values, 2)))
        print('difference between pfasst and lin_pfasst at time %s: %s' %(Tend,np.linalg.norm(u_end_split[-1]-uend.values, np.inf)/np.linalg.norm(
            uex.values, 2)))

    return (max_iter, len(res)-1,it_from_error, min_err, min_err_lin_pfasst, spec_rad, lfa_asymp_conv, cfl)


if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 8

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
    linpparams['tol'] = lparams['restol']


    # fill opts for run with opts
    use_run_method = True
    if use_run_method:
        t0 = 0.0
        dt = 0.1
        results = []
        # dt_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        nu_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        for nu in nu_list:
            pparams['nu'] = nu
            opts = {'description': description, 'linpparams': linpparams, 'tparams': tparams,
                    'mparams': mparams, 'pparams': pparams, 'sparams': sparams, 'lparams': lparams,
                    'num_procs': num_procs, 't0': t0, 'dt': dt}
            results.append(run_with(opts, debug=True))
        for r in results:
            print r
    else:
        pass
        # # quickly generate block of steps
        # MS = mp.generate_steps(num_procs, sparams, description)
        # print "cfl:", pparams['nu']*(pparams['nvars'][0]**2)*dt
        # # get initial values on finest level
        # P = MS[0].levels[0].prob
        # uinit = P.u_exact(t0)
        # # print uinit
        # # call main function to get things done...
        # uend, stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)
        # # print "Type:",type(stats)#,stats
        # # for k,v in stats.items():
        # #     print k.type
        # # print sort_stats(stats, 'type')
        # # MS = mp.generate_steps(num_procs,sparams,description)
        # # u_0 = []
        # # for S,p in zip(MS,range(len(MS))):
        # #     # call predictor from sweeper
        # #     S.status.dt = dt # could have different dt per step here
        # #     S.status.time = t0 + sum(MS[j].status.dt for j in range(p))
        # #     S.init_step(uinit)
        # #     S.levels[0].sweep.predict()
        # #
        # # MS = mp.predictor(MS)
        # #
        # # for S in MS:
        # #     for u in S.u[1:]:
        # #         u_0.append(u)
        #
        #
        # # start with the analysis using the iteration matrix of PFASST
        #
        # transfer_list = mmp.generate_transfer_list(MS, description['transfer_class'], **tparams)
        # lin_pfasst = mmp.generate_LinearPFASST(MS, transfer_list, uinit.values, **tparams)
        # # print lin_pfasst.spectral_radius()
        # # lin_pfasst.check_condition_numbers(p=2)
        # # check the how well the LFA is doing
        # lfa = mmp.LFAForLinearPFASST(lin_pfasst, MS, transfer_list, debug=True)
        # print "lfa:"
        # print lfa.asymptotic_conv_factor()
        # print lin_pfasst.spectral_radius(ka=8, tolerance=1e-7)
        # u_0 = np.kron(np.asarray([1]*description['num_nodes']+[1]*description['num_nodes']*(num_procs-1)),
        #               uinit.values)
        #
        # res, u = mmp.run_linear_pfasst(lin_pfasst, u_0, linpparams)
        # all_nodes = mmp.get_all_nodes(MS, t0)
        # print "Residuals:\n", res, "\nNumber of iterations: ", len(res)-1
        # u_end_split = np.split(u[-1], num_procs*description['num_nodes'])
        #
        #
        # uex = P.u_exact(Tend)
        # print "relative error per linpfasst iteration"
        # for u in u[1:]:
        #     last_u = np.split(u, num_procs*description['num_nodes'])[-1]
        #     print np.linalg.norm(uex.values-last_u, np.inf)/np.linalg.norm(uex.values, np.inf)
        #
        # print('matrix error at time %s: %s' %(Tend, np.linalg.norm(uex.values-u_end_split[-1], np.inf)/np.linalg.norm(
        #     uex.values, 2)))
        # print('non matrix error at time %s: %s' %(Tend,np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(
        #     uex.values, 2)))
        # print('difference between pfasst and lin_pfasst at time %s: %s' %(Tend,np.linalg.norm(u_end_split[-1]-uend.values, np.inf)/np.linalg.norm(
        #     uex.values, 2)))
        # # extract_stats = grep_stats(stats, type='residual')
        # # sortedlist_stats = sort_stats(extract_stats, sortby='step')
        # # for item in sortedlist_stats:
        # #     print(item)
        # # print(extract_stats, sortedlist_stats)
