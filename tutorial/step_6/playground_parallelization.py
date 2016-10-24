import numpy as np
from mpi4py import MPI
from collections import defaultdict


from pySDC.implementations.problem_classes.HeatEquation_1D_FD_forced import heat1d_forced
from pySDC.implementations.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.implementations.controller_classes.allinclusive_classic_MPI import allinclusive_classic_MPI
from pySDC.implementations.controller_classes.allinclusive_multigrid_MPI import allinclusive_multigrid_MPI

from pySDC.plugins.stats_helper import filter_stats, sort_stats


if __name__ == "__main__":
    """
    A simple test program to do PFASST runs for the heat equation
    """

    comm = MPI.COMM_WORLD

    # initialize level parameters
    level_params = {}
    level_params['restol'] = 1E-10
    level_params['dt'] = 0.25

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]
    sweeper_params['do_LU'] = True      # For the IMEX sweeper, the LU-trick can be activated for the implicit part

    # initialize problem parameters
    problem_params = {}
    problem_params['nu'] = 0.1           # diffusion coefficient
    problem_params['freq'] = 8           # frequency for the test value
    problem_params['nvars'] = [511,255]  # number of degrees of freedom for each level

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
    description['problem_class'] = heat1d_forced                    # pass problem class
    description['problem_params'] = problem_params                  # pass problem parameters
    description['dtype_u'] = mesh                                   # pass data type for u
    description['dtype_f'] = rhs_imex_mesh                          # pass data type for f
    description['sweeper_class'] = imex_1st_order                   # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params                  # pass sweeper parameters
    description['level_params'] = level_params                      # pass level parameters
    description['step_params'] = step_params                        # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh              # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params    # pass paramters for spatial transfer

    # set time parameters
    t0 = 0.0
    Tend = 4.0

    # instantiate controller
    controller = allinclusive_classic_MPI(controller_params=controller_params, description=description, comm=comm)
    # controller = allinclusive_multigrid_MPI(controller_params=controller_params, description=description, comm=comm)

    # get initial values on finest level
    P = controller.S.levels[0].prob
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
    print(comm.Get_rank(),iter_counts)

    iter_counts_list = comm.gather(iter_counts, root=0)

    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:

        f = open('step_6_D_out.txt', 'a')
        out = 'Working with %2i processes...' % size
        f.write(out + '\n')
        print(out)

        iter_counts_gather = [item for sublist in iter_counts_list for item in sublist]
        iter_counts = sorted(iter_counts_gather, key=lambda tup: tup[0])
        # compute and print statistics
        for item in iter_counts:
            out = 'Number of iterations for time %4.2f: %2i' % item
            f.write(out + '\n')
            print(out)
        f.write('\n')
        print()
        niters = np.array([item[1] for item in iter_counts])
        out = '   Mean number of iterations: %4.2f' %np.mean(niters)
        f.write(out+'\n')
        print(out)
        out = '   Range of values for number of iterations: %2i ' %np.ptp(niters)
        f.write(out+'\n')
        print(out)
        out = '   Position of max/min number of iterations: %2i -- %2i' %(np.argmax(niters), np.argmin(niters))
        f.write(out+'\n')
        print(out)
        out = '   Std and var for number of iterations: %4.2f -- %4.2f' %(np.std(niters), np.var(niters))
        f.write(out+'\n')
        print(out)

        f.write('\n\n')
        print()
        print()

        assert err < 1.3505E-04, "ERROR: error is too high, got %s" % err
        assert np.ptp(niters) <= 2, "ERROR: range of number of iterations is too high, got %s" % np.ptp(niters)
        assert np.mean(niters) <= 5.94, "ERROR: mean number of iteratiobs is too high, got %s" % np.mean(niters)








