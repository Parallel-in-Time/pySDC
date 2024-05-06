import pytest
import subprocess
import os
import warnings


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
def test_main_parallel():
    # try to import MPI here, will fail if things go wrong (and not in the subprocess part)
    try:
        import mpi4py

        del mpi4py
    except ImportError:
        raise ImportError('petsc tests need mpi4py')

    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

    nprocs = 2
    cmd = f"export PYTHONPATH=$PYTHONPATH:$(pwd); export HWLOC_HIDE_ERRORS=2; mpirun -np {nprocs} python pySDC/projects/AllenCahn_Bayreuth/run_simple_forcing_benchmark.py -n {nprocs}"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
    p.wait()
    (output, err) = p.communicate()
    print(output)
    if err:
        warnings.warn(err)
    # assert err == '', err

    nprocs = 4
    cmd = f"export PYTHONPATH=$PYTHONPATH:$(pwd); export HWLOC_HIDE_ERRORS=2; mpirun -np {nprocs} python pySDC/projects/AllenCahn_Bayreuth/run_simple_forcing_benchmark.py -n {nprocs}"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
    p.wait()
    (output, err) = p.communicate()
    print(output)
    if err:
        warnings.warn(err)
