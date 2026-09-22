import pytest


@pytest.mark.mpi4py
@pytest.mark.parametrize('spectral', [True, False])
@pytest.mark.parametrize('name', ['AC-test-noforce', 'AC-test-constforce', 'AC-test-timeforce'])
def test_main_serial(name, spectral):
    from pySDC.projects.AllenCahn_Bayreuth.run_simple_forcing_verification import run_simulation

    run_simulation(name=name, spectral=spectral, nprocs_space=None)


@pytest.mark.mpi4py
def test_visualize_radii():
    from pySDC.projects.AllenCahn_Bayreuth.run_simple_forcing_verification import visualize_radii

    visualize_radii()


@pytest.mark.slow
@pytest.mark.mpi4py
@pytest.mark.parallel([2, 4])
def test_main_parallel():
    """
    The benchmark has to run on several ranks in space.

    This used to launch two `mpirun`s and only warn on whatever they wrote to stderr -- its one
    assertion was commented out -- so it could not fail. Running the script here means an exception
    in it fails the test.
    """
    from mpi4py import MPI
    from pySDC.projects.AllenCahn_Bayreuth.run_simple_forcing_benchmark import run_simulation

    run_simulation(name='AC-bench-noforce', nprocs_space=MPI.COMM_WORLD.size)
