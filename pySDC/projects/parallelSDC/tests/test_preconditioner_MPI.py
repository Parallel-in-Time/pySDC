import pytest


@pytest.mark.slow
@pytest.mark.mpi4py
@pytest.mark.timeout(600)
@pytest.mark.parallel([3, 5])
def test_preconditioner_playground_MPI():
    """
    The sweeper takes one node per rank, so `num_nodes` is the size of the communicator.

    The plot is produced here rather than in a second process: the docs job collects
    `test-artifacts-*`, so the figure has to exist by the end of this job. That is what the script's
    own `__main__` does with no action argument.

    `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` used to be pinned to 1 in the child environment
    against a threaded BLAS oversubscribing the ranks. Measured on 8 cores, it makes no difference:
    4.97 s against 4.95 s on 3 ranks and 153.1 s against 154.9 s on 5, both inside the noise.
    """
    from mpi4py import MPI
    from pySDC.projects.parallelSDC.preconditioner_playground_MPI import main, plot_iterations

    comm = MPI.COMM_WORLD
    main(comm=comm)

    if comm.rank == 0:
        plot_iterations()
