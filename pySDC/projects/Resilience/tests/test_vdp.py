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
        # `size` is the number of parallel steps, which is the size of the communicator. `vdp.py`'s
        # `__main__` passed `MPI_ready` here, so this ran with `size=1` whatever the rank count.
        check_step_size_limiter(comm.size, comm)


@pytest.mark.mpi4py
def test_adaptivity_with_avoid_restarts():
    from pySDC.projects.Resilience.vdp import check_adaptivity_with_avoid_restarts

    check_adaptivity_with_avoid_restarts(comm=None, size=1)
