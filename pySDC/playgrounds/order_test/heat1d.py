# import pySDC.helpers.plot_helper as plt_helper
#
# import pickle
# import os
import numpy as np

from pySDC.implementations.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_1D_FD_periodic import heat1d_periodic
from pySDC.implementations.problem_classes.HeatEquation_1D_FD_forced import heat1d_forced
from pySDC.implementations.problem_classes.HeatEquation_1D_FD import heat1d
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh

from pySDC.playgrounds.order_test.hook_get_update import get_update

from pySDC.helpers.stats_helper import filter_stats, sort_stats


def main():
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-10
    level_params['dt'] = None

    # This comes as read-in for the step class (this is optional!)
    step_params = dict()
    step_params['maxiter'] = None

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['nu'] = 0.1
    problem_params['freq'] = 2
    problem_params['nvars'] = [2 ** 10 - 1, 2 ** 9 - 1]

    # This comes as read-in for the sweeper class
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [5]
    sweeper_params['QI'] = ['IE']#, 'IE']
    sweeper_params['spread'] = False
    sweeper_params['do_coll_update'] = False

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 6
    space_transfer_params['periodic'] = False

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['predict'] = False
    # controller_params['hook_class'] = get_update

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = heat1d
    description['dtype_u'] = mesh
    description['dtype_f'] = mesh#rhs_imex_mesh
    description['sweeper_class'] = generic_implicit#imex_1st_order
    description['sweeper_params'] = sweeper_params
    description['step_params'] = step_params
    description['level_params'] = level_params
    description['problem_params'] = problem_params
    description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer


    # setup parameters "in time"
    t0 = 0.0
    Tend = 1.0

    dt_list = [Tend / 2 ** i for i in range(0, 8)]
    niter_list = [2]#[1, 2, 3, 4]

    for niter in niter_list:

        err = 0
        for dt in dt_list:

            print('Working with dt = %s and k = %s iterations...' % (dt, niter))

            description['step_params']['maxiter'] = niter
            description['level_params']['dt'] = dt

            # Tend = t0 + dt

            # instantiate the controller
            controller = controller_nonMPI(num_procs=1, controller_params=controller_params,
                                                       description=description)

            # get initial values on finest level
            P = controller.MS[0].levels[0].prob
            uinit = P.u_exact(t0)

            # call main function to get things done...
            uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

            # compute exact solution and compare
            # uex = compute_collocation_solution(controller)
            # print(abs(uex - P.u_exact(Tend)))
            uex = P.u_exact(Tend)
            err_new = abs(uex - uend)

            print('   error at time %s: %s' % (Tend, err_new))
            if err > 0:
                print('   order of accuracy: %6.4f' % (np.log(err / err_new) / np.log(2)))
            # exit()
            err = err_new

            # # filter statistics by type (number of iterations)
            filtered_stats = filter_stats(stats, type='niter')

            # convert filtered statistics to list of iterations count, sorted by process
            iter_counts = sort_stats(filtered_stats, sortby='time')

            # compute and print statistics
            niters = np.array([item[1] for item in iter_counts])
            out = '   Mean number of iterations: %4.2f' % np.mean(niters)
            print(out)
            # out = '   Range of values for number of iterations: %2i ' % np.ptp(niters)
            # # f.write(out + '\n')
            # print(out)
            # out = '   Position of max/min number of iterations: %2i -- %2i' % \
            #       (int(np.argmax(niters)), int(np.argmin(niters)))
            # # f.write(out + '\n')
            # print(out)
            # out = '   Std and var for number of iterations: %4.2f -- %4.2f' % \
            #       (float(np.std(niters)), float(np.var(niters)))
            # # f.write(out + '\n')
            # # f.write(out + '\n')
            # print(out)
        print()


def compute_collocation_solution(controller):

    Q = controller.MS[0].levels[0].sweep.coll.Qmat[1:, 1:]
    A = controller.MS[0].levels[0].prob.A.todense()
    dt = controller.MS[0].levels[0].dt

    N = controller.MS[0].levels[0].prob.init
    M = controller.MS[0].levels[0].sweep.coll.num_nodes

    u0 = np.kron(np.ones(M), controller.MS[0].levels[0].u[0].values)

    C = np.eye(M * N) - dt * np.kron(Q, A)

    tmp = np.linalg.solve(C, u0)
    print(np.linalg.norm(C.dot(tmp) - u0, np.inf))
    uex = controller.MS[0].levels[0].prob.dtype_u(N)
    uex.values[:] = tmp[(M-1) * N:]
    return uex

if __name__ == "__main__":
    main()
