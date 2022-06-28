import numpy as np

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.AdvectionEquation_1D_FD import advection1d
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh


def main():
    """
    A simple test program to run PFASST for the advection equation in multiple ways...
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-09
    level_params['dt'] = 0.0625

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]

    # initialize problem parameters
    problem_params = dict()
    problem_params['c'] = 1  # advection coefficient
    problem_params['freq'] = 4  # frequency for the test value
    problem_params['nvars'] = [128, 64]  # number of degrees of freedom for each level
    problem_params['order'] = 4
    problem_params['type'] = 'upwind'

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 6
    space_transfer_params['periodic'] = True

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['predict_type'] = 'pfasst_burnin'

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = advection1d  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = generic_implicit  # pass sweeper (see part B)
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

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

    f = open('step_5_C_out.txt', 'w')
    # loop over different types of implicit sweeper types
    for QI in QI_list:

        # define and set preconditioner for the implicit sweeper
        sweeper_params['QI'] = QI
        description['sweeper_params'] = sweeper_params  # pass sweeper parameters

        # init min/max iteration counts
        niters_min_all[QI] = 99
        niters_max_all[QI] = 0

        # loop over different number of processes
        for num_proc in num_proc_list:
            out = 'Working with QI = %s on %2i processes...' % (QI, num_proc)
            f.write(out + '\n')
            print(out)
            # instantiate controller
            controller = controller_nonMPI(num_procs=num_proc, controller_params=controller_params,
                                           description=description)

            # get initial values on finest level
            P = controller.MS[0].levels[0].prob
            uinit = P.u_exact(t0)

            # call main function to get things done...
            uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

            # compute exact solution and compare
            uex = P.u_exact(Tend)
            err = abs(uex - uend)

            # filter statistics by type (number of iterations)
            filtered_stats = filter_stats(stats, type='niter')

            # convert filtered statistics to list of iterations count, sorted by process
            iter_counts = sort_stats(filtered_stats, sortby='time')

            # compute and print statistics
            niters = np.array([item[1] for item in iter_counts])
            niters_min_all[QI] = min(np.mean(niters), niters_min_all[QI])
            niters_max_all[QI] = max(np.mean(niters), niters_max_all[QI])
            out = '   Mean number of iterations: %4.2f' % np.mean(niters)
            f.write(out + '\n')
            print(out)
            out = '   Range of values for number of iterations: %2i ' % np.ptp(niters)
            f.write(out + '\n')
            print(out)
            out = '   Position of max/min number of iterations: %2i -- %2i' % \
                  (int(np.argmax(niters)), int(np.argmin(niters)))
            f.write(out + '\n')
            print(out)
            out = '   Std and var for number of iterations: %4.2f -- %4.2f' % \
                  (float(np.std(niters)), float(np.var(niters)))
            f.write(out + '\n')
            f.write(out + '\n')
            print(out)

            f.write('\n')
            print()

            assert err < 5.0716135e-04, "ERROR: error is too high, got %s" % err

        out = 'Mean number of iterations went up from %4.2f to %4.2f for QI = %s!' % \
              (niters_min_all[QI], niters_max_all[QI], QI)
        f.write(out + '\n')
        print(out)

        f.write('\n\n')
        print()
        print()

    f.close()


if __name__ == "__main__":
    main()
