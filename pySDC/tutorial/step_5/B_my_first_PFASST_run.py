from pathlib import Path
import numpy as np

from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_1D_FD_forced import heat1d_forced
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh


def main():
    """
    A simple test program to do PFASST runs for the heat equation
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = 0.25

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['node_type'] = 'LEGENDRE'
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part

    # initialize problem parameters
    problem_params = dict()
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = 8  # frequency for the test value
    problem_params['nvars'] = [511, 255]  # number of degrees of freedom for each level

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 6

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['predict_type'] = 'pfasst_burnin'

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = heat1d_forced  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

    # set time parameters
    t0 = 0.0
    Tend = 4.0

    # set up list of parallel time-steps to run PFASST with
    nsteps = int(Tend / level_params['dt'])
    num_proc_list = [2**i for i in range(int(np.log2(nsteps) + 1))]

    Path("data").mkdir(parents=True, exist_ok=True)
    f = open('data/step_5_B_out.txt', 'w')
    # loop over different number of processes and check results
    for num_proc in num_proc_list:
        out = 'Working with %2i processes...' % num_proc
        f.write(out + '\n')
        print(out)
        # instantiate controller
        controller = controller_nonMPI(num_procs=num_proc, controller_params=controller_params, description=description)

        # get initial values on finest level
        P = controller.MS[0].levels[0].prob
        uinit = P.u_exact(t0)

        # call main function to get things done...
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

        # compute exact solution and compare
        uex = P.u_exact(Tend)
        err = abs(uex - uend)

        # filter statistics by type (number of iterations)
        iter_counts = get_sorted(stats, type='niter', sortby='time')

        # compute and print statistics
        for item in iter_counts:
            out = 'Number of iterations for time %4.2f: %2i' % item
            f.write(out + '\n')
            print(out)
        f.write('\n')
        print()
        niters = np.array([item[1] for item in iter_counts])
        out = '   Mean number of iterations: %4.2f' % np.mean(niters)
        f.write(out + '\n')
        print(out)
        out = '   Range of values for number of iterations: %2i ' % np.ptp(niters)
        f.write(out + '\n')
        print(out)
        out = '   Position of max/min number of iterations: %2i -- %2i' % (
            int(np.argmax(niters)),
            int(np.argmin(niters)),
        )
        f.write(out + '\n')
        print(out)
        out = '   Std and var for number of iterations: %4.2f -- %4.2f' % (float(np.std(niters)), float(np.var(niters)))
        f.write(out + '\n')
        print(out)

        f.write('\n\n')
        print()
        print()

        assert err < 1.3505e-04, "ERROR: error is too high, got %s" % err
        assert np.ptp(niters) <= 1, "ERROR: range of number of iterations is too high, got %s" % np.ptp(niters)
        assert np.mean(niters) <= 5.0, "ERROR: mean number of iteratiobs is too high, got %s" % np.mean(niters)

    f.close()


if __name__ == "__main__":
    main()
