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
        check_step_size_limiter(True, comm)


@pytest.mark.mpi4py
def test_adaptivity_with_avoid_restarts():
    from pySDC.projects.Resilience.vdp import check_adaptivity_with_avoid_restarts

    check_adaptivity_with_avoid_restarts(comm=None, size=1)
