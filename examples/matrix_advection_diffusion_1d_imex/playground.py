
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
from examples.heat1d_periodic.HookClass import error_output

def generate_linpfasst(opts,uinit=False,debug=False):
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
    # initialize the steps like in
    # uend, stats = mp.run_pfasst(MS, u0=uinit, t0=t0, dt=dt, Tend=Tend)

    # initial ordering of the steps: 0,1,...,Np-1
    slots = [p for p in range(opt.num_procs)]

    # initialize time variables of each step
    for p in slots:
        MS[p].status.dt = dt # could have different dt per step here
        MS[p].status.time = t0 + sum(MS[j].status.dt for j in range(p))
        MS[p].status.step = p

    # initialize linear_pfasst
    transfer_list = mmp.generate_transfer_list(MS, opt.description['transfer_class'], **opt.tparams)
    lin_pfasst = mmp.generate_LinearPFASST(MS, transfer_list, uinit.values, **opt.tparams)
#     lfa = mmp.LFAForLinearPFASST(lin_pfasst, MS, transfer_list, debug=True)
#     spec_rad = max(lin_pfasst.spectral_radius(ka=8, tolerance=1e-7))
#     lfa_asymp_conv = lfa.asymptotic_conv_factor()
    return lin_pfasst,MS,transfer_list

if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 5

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-10

    sparams = {}
    sparams['maxiter'] = 35

    # This comes as read-in for the problem class
    pparams = {}
    pparams['c'] = 1.0
    pparams['nu'] = 0.1
    pparams['t_0'] = 0.1
    pparams['nvars'] = [64, 32]
    pparams['order'] = [6]

    # This comes as read-in for the all kind of generating options for the matrix classes
    mparams = {}
    mparams['sparse_format'] = "array"

    # This comes as read-in for the transfer operations
    tparams = {}
    tparams['finter'] = False
    tparams['sparse_format'] = "array"
    tparams['iorder'] = 2
    tparams['rorder'] = 2
    tparams['interpolation_order'] = [2]*num_procs
    tparams['restriction_order'] = [2]*num_procs
    tparams['t_interpolation_order'] = [2]*num_procs
    tparams['t_restriction_order'] = [2]*num_procs
#     tparams['q_precond'] = 'QD'
    tparams['q_precond'] = 'QI'

    swparams = {}
    swparams['do_LU'] = True

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = advection_diffusion
    description['problem_params'] = pparams
    description['dtype_u'] = mesh
    description['dtype_f'] = rhs_imex_mesh
    description['collocation_class'] = collclass.CollGaussRadau_Right
    description['num_nodes'] = 5
    description['sweeper_class'] = imex_1st_order
    description['level_params'] = lparams
    description['transfer_class'] = mesh_to_mesh_1d_periodic
    description['transfer_params'] = tparams
    description['sweeper_params'] = swparams
    description['hook_class'] = error_output


    # Options for run_linear_pfasst
    linpparams = {}
    linpparams['run_type'] = "tolerance"
    linpparams['norm'] = lambda x: np.linalg.norm(x, np.inf)
    linpparams['tol'] = lparams['restol']*1

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs,sparams,description)

    # setup parameters "in time"
    # initial time
    t0 = 0.0
    dt = 0.1
    Tend = num_procs*dt
    dx = 1.0/pparams['nvars'][0]
    # eigenvalues of the periodic and dirichlet case

    def laplace_eigvalues_periodic(n,k):
        return +2*np.cos(2*np.pi/n*k)-2.0

    def laplace_eigenvalues_dirichlet(n,k):
        return 4.0*np.sin(k*np.pi/(2*n))**2

    opts = {'description': description, 'linpparams': linpparams, 'tparams': tparams,
            'mparams': mparams, 'pparams': pparams, 'sparams': sparams, 'lparams': lparams,
            'num_procs': num_procs, 't0': t0, 'dt': dt}

    def solution_laplace_periodic(x,t,n,nu=pparams['nu'],L=1.0):
        return np.exp(-nu*(2*np.pi*n/L)**2*t)*np.sin((2*np.pi*n/L)*x)
    opt = mmp.Bunch()
    for k, v in opts.items():
        setattr(opt, k, v)

    lin_pfasst, MS, transfer_list = generate_linpfasst(opts)
    # get initial values on finest level
    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    # print "initial values"
    # print uinit.values

    # call main function to get things done...
    # uend,stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    # start with the analysis using the iteration matrix of PFASST

    transfer_list = mmp.generate_transfer_list(MS, description['transfer_class'], **tparams)
    lin_pfasst = mmp.generate_LinearPFASST(MS, transfer_list, uinit.values, **tparams)
    # lin_pfasst.check_condition_numbers(p=2)
    u_0 = np.kron(np.asarray([1]*description['num_nodes']+[0]*description['num_nodes']*(num_procs-1)),
                  uinit.values)
    res, u = mmp.run_linear_pfasst(lin_pfasst, u_0, linpparams)
    all_nodes = mmp.get_all_nodes(MS, t0)
    print "Residuals:\n", res, "\nNumber of iterations: ", len(res)-1
    u_end_split = np.split(u[-1], num_procs*description['num_nodes'])
    # print "Spectral Radius:\t", lin_pfasst.spectral_radius()
    # lfa = mmp.LFAForLinearPFASST(lin_pfasst, MS, transfer_list, debug=True)
    # print "lfa:"
    # print lfa.asymptotic_conv_factor()
    # res, u = mmp.run_linear_pfasst(lin_pfasst, u_0, linpparams)

    # compute exact solution and compare
    uex = P.u_exact(Tend)
    print "relative error per linpfasst iteration"
    for u in u[1:]:
        last_u = np.split(u, num_procs*description['num_nodes'])[-1]
        print np.linalg.norm(uex.values-last_u, np.inf)/np.linalg.norm(uex.values, np.inf)

    # print('error at time %s: %s' %(Tend,np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(
    #     uex.values,np.inf)))
    print('matrix error at time %s: %s' %(Tend,np.linalg.norm(uex.values-u_end_split[-1], np.inf)/np.linalg.norm(
        uex.values,np.inf)))

    # print(' absolute difference between pfasst and lin_pfasst at time %s: %s' %(Tend,np.linalg.norm(u_end_split[-1]-uend.values, np.inf)))
    # print(u_end_split[-1])
    # print(uend.values)
    # print(uend.values - u_end_split[-1])
    # print(uex.values - uend.values)
    # print(uinit.values - uex.values)
    # extract_stats = grep_stats(stats,iter=-1,type='residual')
    # sortedlist_stats = sort_stats(extract_stats,sortby='step')
    # print(extract_stats,sortedlist_stats)
