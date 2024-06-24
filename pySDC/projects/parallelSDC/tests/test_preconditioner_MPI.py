import os
import subprocess
import pytest


@pytest.mark.slow
@pytest.mark.mpi4py
@pytest.mark.timeout(600)
@pytest.mark.parametrize('num_procs', [3, 5])
def test_preconditioner_playground_MPI(num_procs):

    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'
    my_env['OPENBLAS_NUM_THREADS'] = '1'
    my_env['MKL_NUM_THREADS'] = '1'
    cwd = '.'
    cmd = (
        'mpirun -np ' + str(num_procs) + ' python -u pySDC/projects/parallelSDC/preconditioner_playground_MPI.py'
    ).split()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env, cwd=cwd)
    p.wait()
    for line in p.stdout:
        print(line)
    for line in p.stderr:
        print(line)
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (p.returncode, num_procs)
