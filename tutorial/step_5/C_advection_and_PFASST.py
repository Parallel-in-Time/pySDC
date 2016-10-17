import numpy as np

from pySDC.implementations.problem_classes.AdvectionEquation_1D_FD import advection1d
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.TransferMesh_1D import mesh_to_mesh_1d_periodic
from pySDC.implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI

from pySDC.plugins.stats_helper import filter_stats, sort_stats


def main():
    """
    A simple test program to run PFASST for the advection equation in multiple ways...
    """

    # initialize level parameters
    level_params = {}
    level_params['restol'] = 1E-09
    level_params['dt'] = 0.0625

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]

    # initialize problem parameters
    problem_params = {}
    problem_params['c'] = 1             # advection coefficient
    problem_params['freq'] = 4          # frequency for the test value
    problem_params['nvars'] = [128,64]  # number of degrees of freedom for each level
    problem_params['order'] = 4
    problem_params['type'] = 'upwind'

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 50

    # initialize space transfer parameters
    space_transfer_params = {}
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 6

    # initialize controller parameters
    controller_params = {}
    controller_params['logger_level'] = 30

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = advection1d                    # pass problem class
    description['problem_params'] = problem_params                  # pass problem parameters
    description['dtype_u'] = mesh                                   # pass data type for u
    description['dtype_f'] = mesh                                  # pass data type for f
    description['sweeper_class'] = generic_implicit                  # pass sweeper (see part B)
    description['level_params'] = level_params                      # pass level parameters
    description['step_params'] = step_params                        # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh_1d_periodic # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params    # pass paramters for spatial transfer

    # set time parameters
    t0 = 0.0
    Tend = 1.0

    # set up list of parallel time-steps to run PFASST with
    nsteps = int(Tend / level_params['dt'])
    num_proc_list = [2 ** i for i in range(int(np.log2(nsteps) + 1))]

    # set up list of types of implicit SDC sweepers: LU and implicit Euler here
    QI_list = ['LU', 'IE']
    niters_min_all = {}
    niters_max_all = {}

    for QI in QI_list:

        # define and set preconditioner for the implicit sweeper
        sweeper_params['QI'] = QI
        description['sweeper_params'] = sweeper_params  # pass sweeper parameters

        # init min/max iteration counts
        niters_min_all[QI] = 99
        niters_max_all[QI] = 0

        for num_proc in num_proc_list:
            print('Working with QI = %s on %2i processes...' %(QI,num_proc))
            # instantiate controller
            controller = allinclusive_classic_nonMPI(num_procs=num_proc, controller_params=controller_params,
                                                     description=description)

            # get initial values on finest level
            P = controller.MS[0].levels[0].prob
            uinit = P.u_exact(t0)

            # call main function to get things done...
            uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

            # compute exact solution and compare
            uex = P.u_exact(Tend)
            err = abs(uex - uend)
            print('Error: %12.8e' % (err))

            # filter statistics by type (number of iterations)
            filtered_stats = filter_stats(stats, type='niter')

            # convert filtered statistics to list of iterations count, sorted by process
            iter_counts = sort_stats(filtered_stats, sortby='time')

            # compute and print statistics
            niters = np.array([item[1] for item in iter_counts])
            niters_min_all[QI] = min(np.mean(niters),niters_min_all[QI])
            niters_max_all[QI] = max(np.mean(niters), niters_max_all[QI])
            print('   Mean number of iterations: %4.2f' % np.mean(niters))
            print('   Range of values for number of iterations: %2i ' % np.ptp(niters))
            print('   Position of max/min number of iterations: %2i -- %2i' % (np.argmax(niters), np.argmin(niters)))
            print('   Std and var for number of iterations: %4.2f -- %4.2f' % (np.std(niters), np.var(niters)))
            print()

            assert err < 5.0716135e-04, "ERROR: error is too high, got %s" % err

        print('Mean number of iterations went up from %4.2f to %4.2f for QI = %s!' %(niters_min_all[QI], niters_max_all[QI], QI))
        print()
        print()

if __name__ == "__main__":
    main()
