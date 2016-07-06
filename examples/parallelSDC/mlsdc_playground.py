from pySDC import CollocationClasses as collclass

import numpy as np

from examples.heat1d_unforced.ProblemClass import heat1d_unforced
from examples.heat1d_unforced.TransferClass import mesh_to_mesh_1d
from examples.advection_1d_implicit.ProblemClass import advection
from examples.advection_1d_implicit.TransferClass import mesh_to_mesh_1d_periodic
from pySDC.datatype_classes.mesh import mesh


from pySDC.sweeper_classes.generic_implicit import generic_implicit
import pySDC.PFASST_blockwise as mp

from pySDC import Log
from pySDC.Stats import grep_stats, sort_stats

from pySDC.Plugins.sweeper_helper import get_Qd

from collections import namedtuple
import pickle

if __name__ == "__main__":

    ID = namedtuple('ID', ['setup', 'qd_type', 'param'])

    # set global logger (remove this if you do not want the output at all)
    # logger = Log.setup_custom_logger('root')

    num_procs = 1

    # This comes as read-in for the level class  (this is optional!)
    lparams = {}
    lparams['restol'] = 1E-10

    # This comes as read-in for the step class (this is optional!)
    sparams = {}
    sparams['maxiter'] = 100
    sparams['fine_comm'] = True

    # This comes as read-in for the transfer operations (this is optional!)
    tparams = {}
    tparams['finter'] = False
    tparams['iorder'] = 2
    tparams['rorder'] = 2

    Nnodes = 3
    cclass = collclass.CollGaussRadau_Right
    dt = 0.1

    qd_list = [ 'LU', 'IE', ('LU','LU'), ('IE','IE'), ('IEpar','IE'), ('IEpar','LU'), ('LU','IEpar'), ('IE','IEpar') ]
                # ('LU','Qpar'), ('IE','Qpar'), ('Qpar', 'IE'), ('Qpar', 'LU'), ('PIC','IE'), ('PIC','LU') ]

    setup_list = [ ('heat',(63, 31), [10.0**i for i in range(-3,3)]),
                   ('advection',(64, 32), [10.0**i for i in range(-3,3)]) ]

    results = {}
    for setup, nvars, param_list in setup_list:
        results[setup] = (nvars,param_list)

    for qd_type in qd_list:

        # This comes as read-in for the sweeper class
        swparams = {}
        if not type(qd_type) == str:
            swparams['QI'] = get_Qd(cclass, Nnodes=Nnodes, qd_type=qd_type[0])
            swparams_c = {}
            swparams_c['QI'] = get_Qd(cclass, Nnodes=Nnodes, qd_type=qd_type[1])
        else:
            swparams['QI'] = get_Qd(cclass, Nnodes=Nnodes, qd_type=qd_type)

        for setup, nvars, param_list in setup_list:

            pparams = {}
            if not type(qd_type) == str:
                pparams['nvars'] = [nvars[0], nvars[1]]
            else:
                pparams['nvars'] = nvars[0]

            for param in param_list:

                description = {}
                description['dtype_u'] = mesh
                description['dtype_f'] = mesh
                description['collocation_class'] = cclass
                description['num_nodes'] = Nnodes
                description['sweeper_class'] = generic_implicit
                if not type(qd_type) == str:
                    description['sweeper_params'] = [swparams, swparams_c]
                else:
                    description['sweeper_params'] = swparams
                description['level_params'] = lparams

                print('working on: %s - %s - %s' % (qd_type, setup, param))

                if setup == 'heat':

                    pparams['nu'] = param
                    pparams['k'] = 2
                    description['problem_class'] = heat1d_unforced
                    description['transfer_class'] = mesh_to_mesh_1d
                    dt = 0.1

                elif setup == 'advection':

                    pparams['c'] = param
                    pparams['order'] = 2
                    description['problem_class'] = advection
                    description['transfer_class'] = mesh_to_mesh_1d_periodic
                    dt = 0.1


                else:
                    print('Setup not implemented..',setup)
                    exit()

                description['transfer_params'] = tparams
                description['problem_params'] = pparams

                # quickly generate block of steps
                MS = mp.generate_steps(num_procs,sparams,description)

                # get initial values on finest level
                P = MS[0].levels[0].prob
                uinit = P.u_exact(0)

                # call main function to get things done...
                uend,stats = mp.run_pfasst(MS,u0=uinit,t0=0,dt=dt,Tend=dt)

                extract_stats = grep_stats(stats, level=-1, type='niter')
                sortedlist_stats = sort_stats(extract_stats,sortby='step')
                niter = sortedlist_stats[0][1]
                # compute exact solution and compare
                # uex = P.u_exact(dt)
                # print('error at time %s: %s' %(dt,np.linalg.norm(uex.values-uend.values,np.inf)))
                id = ID(setup=setup, qd_type=qd_type, param=param)
                results[id] = niter

    file = open('results_iterations_mlsdc.pkl', 'wb')
    pickle.dump(results, file)