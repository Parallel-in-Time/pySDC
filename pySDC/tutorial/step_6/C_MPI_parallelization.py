import sys
from pathlib import Path

from mpi4py import MPI

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
from pySDC.tutorial.step_6.A_run_non_MPI_controller import set_parameters_ml


def main(fname='step_6_C_out.txt'):
    """
    Run PFASST with the MPI-parallel controller, one step per rank.

    Run it the way you would run any MPI program, with as many ranks as you want steps::

        mpirun -np 4 python C_MPI_parallelization.py

    The number of parallel steps is simply the size of ``MPI.COMM_WORLD``, so nothing in here has to
    be told how many there are.

    Args:
        fname (str): file under ``data/`` to append the results to
    """

    # set MPI communicator
    comm = MPI.COMM_WORLD

    # get parameters from Part A
    description, controller_params, t0, Tend = set_parameters_ml()

    # instantiate controllers
    controller = controller_MPI(controller_params=controller_params, description=description, comm=comm)
    # get initial values on finest level
    P = controller.S.levels[0].prob
    uinit = P.u_exact(t0)

    # call main functions to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # filter statistics by type (number of iterations)
    iter_counts = get_sorted(stats, type='niter', sortby='time')

    # combine statistics into list of statistics
    iter_counts_list = comm.gather(iter_counts, root=0)

    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        Path("data").mkdir(parents=True, exist_ok=True)
        f = open('data/' + fname, 'a')
        out = 'Working with %2i processes...' % size
        f.write(out + '\n')
        print(out)

        # compute exact solutions and compare with both results
        uex = P.u_exact(Tend)
        err = abs(uex - uend)

        out = 'Error vs. exact solution: %12.8e' % err
        f.write(out + '\n')
        print(out)

        # build one list of statistics instead of list of lists, the sort by time
        iter_counts_gather = [item for sublist in iter_counts_list for item in sublist]
        iter_counts = sorted(iter_counts_gather, key=lambda tup: tup[0])

        # compute and print statistics
        for item in iter_counts:
            out = 'Number of iterations for time %4.2f: %1i ' % (item[0], item[1])
            f.write(out + '\n')
            print(out)

        f.write('\n')
        print()

        assert all(item[1] <= 8 for item in iter_counts), "ERROR: weird iteration counts, got %s" % iter_counts


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) == 2 else 'step_6_C_out.txt')
