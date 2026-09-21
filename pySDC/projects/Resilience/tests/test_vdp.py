import pytest


@pytest.mark.mpi4py
@pytest.mark.parallel([1, 2, 5, 8])
@pytest.mark.parametrize('test_name', ['mpi_vs_nonMPI', 'check_step_size_limiter'])
def test_stuff(test_name):
    from mpi4py import MPI
    from pySDC.projects.Resilience.vdp import mpi_vs_nonMPI, check_step_size_limiter

    comm = MPI.COMM_WORLD
    if test_name == 'mpi_vs_nonMPI':
        mpi_vs_nonMPI(True, comm)
    else:
        # `size` only controls how many steps are trimmed from each end of the step-size window
        # before the limits are checked -- `num_procs` is ignored once a communicator is passed.
        # Deliberately 1 rather than `comm.size`: it checks every step except the first and last,
        # which is stricter than the function's own default and passes at every rank count here.
        # `vdp.py`'s `__main__` passed `MPI_ready`, which is how this came to be 1 by accident.
        check_step_size_limiter(1, comm)


@pytest.mark.mpi4py
def test_adaptivity_with_avoid_restarts():
    from pySDC.projects.Resilience.vdp import check_adaptivity_with_avoid_restarts

    check_adaptivity_with_avoid_restarts(comm=None, size=1)
