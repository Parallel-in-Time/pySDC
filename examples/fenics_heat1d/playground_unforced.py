import pickle
from collections import namedtuple

import dolfin as df

from pySDC.controller_classes.allinclusive_multigrid_nonMPI import allinclusive_multigrid_nonMPI
from examples.fenics_heat1d.ProblemClass_unforced import fenics_heat_unforced
from examples.fenics_heat1d.TransferClass import mesh_to_mesh_fenics
from pySDC import CollocationClasses as collclass
from pySDC.Stats import grep_stats, sort_stats
from pySDC.datatype_classes.fenics_mesh import fenics_mesh
from pySDC.sweeper_classes.generic_implicit import generic_implicit

if __name__ == "__main__":

    ID = namedtuple('ID', ['c_nvars', 'dt'])

    # set global logger (remove this if you do not want the output at all)
    # logger = Log.setup_custom_logger('root')

    num_procs = 5

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
    swparams['QI'] = 'LU'

    # This comes as read-in for the transfer operations
    tparams = {}
    tparams['finter'] = False

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nu'] = 0.1
    pparams['k'] = 1
    pparams['family'] = 'CG'
    pparams['order'] = [1]
    pparams['refinements'] = [1,0]

    # setup parameters "in time"
    t0 = 0.0
    Tend = 0.4

    # dt_list = [Tend/(2**i) for i in range(0,7,1)]
    # c_nvars_list = [2**i for i in range(2,7)]
    dt_list = [0.1]
    c_nvars_list = [256]

    results = {}
    # results['description'] = (pparams,swparams)

    for c_nvars in c_nvars_list:

        pparams['c_nvars'] = c_nvars

        # Fill description dictionary for easy hierarchy creation
        description = {}
        description['problem_class'] = fenics_heat_unforced
        description['problem_params'] = pparams
        description['dtype_u'] = fenics_mesh
        description['dtype_f'] = fenics_mesh
        description['sweeper_class'] = generic_implicit
        description['sweeper_params'] = swparams
        description['level_params'] = lparams
        description['transfer_class'] = mesh_to_mesh_fenics
        description['transfer_params'] = tparams

        # initialize controller
        PFASST = allinclusive_multigrid_nonMPI(num_procs=num_procs, step_params=sparams, description=description)

        # get initial values on finest level
        P = PFASST.MS[0].levels[0].prob
        uinit = P.u_exact(t0)

        # compute exact solution to compare
        uex = P.u_exact(Tend)

        for dt in dt_list:

            print('Working on: c_nvars = %s, dt = %s' %(c_nvars,dt))

            # call main function to get things done...
            uend, stats = PFASST.run(u0=uinit, t0=t0, dt=dt, Tend=Tend)

            err_classical_rel = abs(uex-uend)/abs(uex)
            err_fenics = df.errornorm(uex.values,uend.values)

            print('(classical/fenics) error at time %s: %s / %s' % (Tend,err_classical_rel,err_fenics))

            extract_stats = grep_stats(stats, level=-1, type='niter')
            sortedlist_stats = sort_stats(extract_stats, sortby='step')
            niter = sortedlist_stats[0][1]
            print('niter = %s' %niter)

            id = ID(c_nvars=c_nvars, dt=dt)
            results[id] = (niter,err_classical_rel,err_fenics)
            file = open('fenics_heat_unforced_mlsdc_CG2.pkl', 'wb')
            pickle.dump(results, file)

    print(results)
    # file = open('fenics_heat_unforced_sdc.pkl', 'wb')
    # pickle.dump(results, file)









