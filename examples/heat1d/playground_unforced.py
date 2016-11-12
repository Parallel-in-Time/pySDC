import pickle
from collections import namedtuple

import pySDC.core.deprecated.PFASST_blockwise_old as mp

from pySDC.implementations.problem_classes.HeatEquation_1D_FD import heat1d
from examples.heat1d.TransferClass import mesh_to_mesh_1d
from pySDC.implementations.datatype_classes import mesh
from pySDC.implementations.sweeper_classes.generic_LU import generic_LU
from pySDC.core import CollocationClasses as collclass
from pySDC.core.Stats import grep_stats, sort_stats

if __name__ == "__main__":

    ID = namedtuple('ID', ['nvars', 'dt'])

    # set global logger (remove this if you do not want the output at all)
    # logger = Log.setup_custom_logger('root')

    num_procs = 1

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 5E-09

    # This comes as read-in for the steps
    sparams = {}
    sparams['maxiter'] = 20

    # This comes as read-in for the sweeper class
    swparams = {}
    swparams['collocation_class'] = collclass.CollGaussRadau_Right
    swparams['num_nodes'] = 3

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nu'] = 0.1
    pparams['k'] = 1

    # This comes as read-in for the transfer operations (this is optional!)
    tparams = {}
    tparams['finter'] = False
    tparams['iorder'] = 6
    tparams['rorder'] = 2

    # setup parameters "in time"
    t0 = 0.0
    Tend = 2.0

    # dt_list = [Tend/(2**i) for i in range(0,7,1)]
    # nvars_list = [2**i-1 for i in range(8,14)]
    # nvars_list = [[2**i-1, 2**(i-1)-1] for i in range(8, 14)]
    dt_list = [2.0,1.0]
    nvars_list = [511]

    results = {}
    # results['description'] = (pparams,swparams)

    for nvars in nvars_list:

        pparams['nvars'] = nvars

        # Fill description dictionary for easy hierarchy creation
        description = {}
        description['problem_class'] = heat1d
        description['problem_params'] = pparams
        description['dtype_u'] = mesh
        description['dtype_f'] = mesh
        description['sweeper_class'] = generic_LU
        description['sweeper_params'] = swparams
        description['level_params'] = lparams
        description['transfer_class'] = mesh_to_mesh_1d
        description['transfer_params'] = tparams

        # quickly generate block of steps
        MS = mp.generate_steps(num_procs, sparams, description)

        # get initial values on finest level
        P = MS[0].levels[0].prob
        uinit = P.u_exact(t0)

        # compute exact solution to compare
        uex = P.u_exact(Tend)

        for dt in dt_list:

            print('Working on: c_nvars = %s, dt = %s' %(nvars,dt))

            # call main function to get things done...
            uend, stats = mp.run_pfasst(MS, u0=uinit, t0=t0, dt=dt, Tend=Tend)

            err_rel = abs(uex-uend)/abs(uex)

            print('error at time %s: %s' % (Tend,err_rel))

            extract_stats = grep_stats(stats, level=-1, type='niter')
            sortedlist_stats = sort_stats(extract_stats, sortby='step')
            niter = sortedlist_stats[0][1]

            if type(nvars) is list:
                id = ID(nvars=nvars[0], dt=dt)
            else:
                id = ID(nvars=nvars, dt=dt)
            results[id] = (niter,err_rel)
            # file = open('fd_heat_unforced_sdc.pkl', 'wb')
            file = open('fd_heat_unforced_mlsdc.pkl', 'wb')
            pickle.dump(results, file)

    print(results)
    # file = open('fenics_heat_unforced_sdc.pkl', 'wb')
    # pickle.dump(results, file)









