import numpy as np
from mpi4py import MPI
from collections import defaultdict


from pySDC.implementations.problem_classes.HeatEquation_1D_FD import heat1d
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.sweeper_classes.generic_LU import generic_LU
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
    level_params['restol'] = 5E-10
    level_params['dt'] = 0.125

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]

    # initialize problem parameters
    problem_params = {}
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = 2  # frequency for the test value
    problem_params['nvars'] = [63,31]  # number of degrees of freedom for each level

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
    controller_params['predict'] = False

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = heat1d  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['dtype_u'] = mesh  # pass data type for u
    description['dtype_f'] = mesh  # pass data type for f
    description['sweeper_class'] = generic_LU  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

    # set time parameters
    t0 = 0.0
    Tend = 1.0

    # instantiate controller
    controller_classic = allinclusive_classic_MPI(controller_params=controller_params, description=description, comm=comm)

    # get initial values on finest level
    P = controller_classic.S.levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend_classic, stats_classic = controller_classic.run(u0=uinit, t0=t0, Tend=Tend)

    controller_multigrid = allinclusive_multigrid_MPI(controller_params=controller_params, description=description,
                                                      comm=comm)
    # get initial values on finest level
    P = controller_multigrid.S.levels[0].prob
    uinit = P.u_exact(t0)

    uend_multigrid, stats_multigrid = controller_multigrid.run(u0=uinit, t0=t0, Tend=Tend)

    # filter statistics by type (number of iterations)
    filtered_stats_classic = filter_stats(stats_classic, type='niter')
    filtered_stats_multigrid = filter_stats(stats_multigrid, type='niter')

    # convert filtered statistics to list of iterations count, sorted by process
    iter_counts_classic = sort_stats(filtered_stats_classic, sortby='time')
    iter_counts_multigrid = sort_stats(filtered_stats_multigrid, sortby='time')

    iter_counts_classic_list = comm.gather(iter_counts_classic, root=0)
    iter_counts_multigrid_list = comm.gather(iter_counts_multigrid, root=0)


    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:

        f = open('step_6_B_out.txt', 'a')

        # compute exact solution and compare
        uex = P.u_exact(Tend)
        err_classic = abs(uex - uend_classic)
        err_multigrid = abs(uex - uend_multigrid)
        diff = abs(uend_classic - uend_multigrid)

        out = 'Error classic: %12.8e' % (err_classic)
        f.write(out + '\n')
        print(out)
        out = 'Error multigrid: %12.8e' % (err_multigrid)
        f.write(out + '\n')
        print(out)
        out = 'Diff: %12.8e' % diff
        f.write(out + '\n')
        print(out)

        # compute and print statistics
        iter_counts_gather = [item for sublist in iter_counts_classic_list for item in sublist]
        iter_counts_classic = sorted(iter_counts_gather, key=lambda tup: tup[0])
        iter_counts_gather = [item for sublist in iter_counts_multigrid_list for item in sublist]
        iter_counts_multigrid = sorted(iter_counts_gather, key=lambda tup: tup[0])

        for item_classic, item_multigrid in zip(iter_counts_classic, iter_counts_multigrid):
            out = 'Number of iterations for time %4.2f (classic/multigrid): %1i / %1i' % (
            item_classic[0], item_classic[1], item_multigrid[1])
            f.write(out + '\n')
            print(out)

        f.write('\n')
        print()

        assert all([item[1] <= 7 for item in
                    iter_counts_multigrid]), "ERROR: weird iteration counts for multigrid, got %s" % iter_counts_multigrid
        assert diff < 1.308E-09, "ERROR: difference between classic and multigrid controller is too large, got %s" % diff








