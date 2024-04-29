import os
import subprocess
import pytest


@pytest.mark.slow
@pytest.mark.mpi4py
@pytest.mark.timeout(0)
def test_preconditioner_playground_MPI_5():
    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'
    cwd = '.'
    num_procs = 5
    cmd = (
        'mpirun -np ' + str(num_procs) + ' python pySDC/projects/parallelSDC/preconditioner_playground_MPI.py'
    ).split()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env, cwd=cwd)
    p.wait()
    for line in p.stdout:
        print(line)
    for line in p.stderr:
        print(line)
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (p.returncode, num_procs)


@pytest.mark.slow
@pytest.mark.mpi4py
@pytest.mark.timeout(0)
def test_preconditioner_playground_MPI_3():
    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'
    cwd = '.'
    num_procs = 3
    cmd = (
        'mpirun -np ' + str(num_procs) + ' python pySDC/projects/parallelSDC/preconditioner_playground_MPI.py'
    ).split()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env, cwd=cwd)
    p.wait()
    for line in p.stdout:
        print(line)
    for line in p.stderr:
        print(line)
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (p.returncode, num_procs)
