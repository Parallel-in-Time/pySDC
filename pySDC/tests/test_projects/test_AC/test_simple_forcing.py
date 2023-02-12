import pytest
import subprocess
import os


@pytest.mark.mpi4py
def test_main_serial():
    from pySDC.projects.AllenCahn_Bayreuth.run_simple_forcing_verification import main, visualize_radii

    main()
    visualize_radii()


@pytest.mark.slow
@pytest.mark.mpi4py
def test_main_parallel():
    # try to import MPI here, will fail if things go wrong (and not later on in the subprocess part)
    import mpi4py

    del mpi4py

    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

    nprocs = 2
    cmd = f"export PYTHONPATH=$PYTHONPATH:$(pwd); export HWLOC_HIDE_ERRORS=2; mpirun -np {nprocs} python pySDC/projects/AllenCahn_Bayreuth/run_simple_forcing_benchmark.py -n {nprocs}"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
    p.wait()
    (output, err) = p.communicate()
    print(output)
    assert err == '', err

    nprocs = 4
    cmd = f"export PYTHONPATH=$PYTHONPATH:$(pwd); export HWLOC_HIDE_ERRORS=2; mpirun -np {nprocs} python pySDC/projects/AllenCahn_Bayreuth/run_simple_forcing_benchmark.py -n {nprocs}"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
    p.wait()
    (output, err) = p.communicate()
    print(output)
    assert err == '', err
