from mpi4py import MPI
import sys

from pySDC.implementations.controller_classes.allinclusive_classic_MPI import allinclusive_classic_MPI
from pySDC.implementations.controller_classes.allinclusive_multigrid_MPI import allinclusive_multigrid_MPI

from pySDC.plugins.stats_helper import filter_stats, sort_stats
from tutorial.step_6.A_classic_vs_multigrid_controller import set_parameters

if __name__ == "__main__":
    """
    A simple test program to do MPI-parallel PFASST runs
    """

    # set MPI communicator
    comm = MPI.COMM_WORLD

    # get parameters from Part A
    description, controller_params, t0, Tend = set_parameters()

    # instantiate controllers
    controller_classic = allinclusive_classic_MPI(controller_params=controller_params, description=description,
                                                  comm=comm)
    controller_multigrid = allinclusive_multigrid_MPI(controller_params=controller_params, description=description,
                                                      comm=comm)
    # get initial values on finest level
    P = controller_classic.S.levels[0].prob
    uinit = P.u_exact(t0)

    # call main functions to get things done...
    uend_classic, stats_classic = controller_classic.run(u0=uinit, t0=t0, Tend=Tend)
    uend_multigrid, stats_multigrid = controller_multigrid.run(u0=uinit, t0=t0, Tend=Tend)

    # filter statistics by type (number of iterations)
    filtered_stats_classic = filter_stats(stats_classic, type='niter')
    filtered_stats_multigrid = filter_stats(stats_multigrid, type='niter')

    # convert filtered statistics to list of iterations count, sorted by process
    iter_counts_classic = sort_stats(filtered_stats_classic, sortby='time')
    iter_counts_multigrid = sort_stats(filtered_stats_multigrid, sortby='time')

    # combine statistics into list of statistics
    iter_counts_classic_list = comm.gather(iter_counts_classic, root=0)
    iter_counts_multigrid_list = comm.gather(iter_counts_multigrid, root=0)

    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:

        # we'd need to deal with variable file names here (for testign purpose only)
        if len(sys.argv) == 2:
            fname = sys.argv[1]
        else:
            fname = 'step_6_B_out.txt'

        f = open(fname, 'a')
        out = 'Working with %2i processes...' % size
        f.write(out + '\n')
        print(out)

        # compute exact solutions and compare with both results
        uex = P.u_exact(Tend)
        err_classic = abs(uex - uend_classic)
        err_multigrid = abs(uex - uend_multigrid)
        diff = abs(uend_classic - uend_multigrid)

        out = 'Error classic: %12.8e' % err_classic
        f.write(out + '\n')
        print(out)
        out = 'Error multigrid: %12.8e' % err_multigrid
        f.write(out + '\n')
        print(out)
        out = 'Diff: %6.4e' % diff
        f.write(out + '\n')
        print(out)

        # build one list of statistics instead of list of lists, the sort by time
        iter_counts_gather = [item for sublist in iter_counts_classic_list for item in sublist]
        iter_counts_classic = sorted(iter_counts_gather, key=lambda tup: tup[0])
        iter_counts_gather = [item for sublist in iter_counts_multigrid_list for item in sublist]
        iter_counts_multigrid = sorted(iter_counts_gather, key=lambda tup: tup[0])

        # compute and print statistics
        for item_classic, item_multigrid in zip(iter_counts_classic, iter_counts_multigrid):
            out = 'Number of iterations for time %4.2f (classic/multigrid): %1i / %1i' % \
                  (item_classic[0], item_classic[1], item_multigrid[1])
            f.write(out + '\n')
            print(out)

        f.write('\n')
        print()

        assert all([item[1] <= 7 for item in iter_counts_multigrid]), \
            "ERROR: weird iteration counts for multigrid, got %s" % iter_counts_multigrid
        assert diff < 1.308E-09, \
            "ERROR: difference between classic and multigrid controller is too large, got %s" % diff
