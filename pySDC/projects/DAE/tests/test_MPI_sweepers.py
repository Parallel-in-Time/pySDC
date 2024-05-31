import pytest


@pytest.mark.mpi4py
@pytest.mark.parametrize('num_procs', [2, 3])
def testOrder(num_procs):
    r"""
    Test checks if order of accuracy is reached for the MPI sweepers.
    """

    import os
    import subprocess

    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'
    cwd = '.'

    cmd = ('mpirun -np ' + str(num_procs) + ' python pySDC/projects/DAE/run/accuracy_check_MPI.py').split()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env, cwd=cwd)

    p.wait()
    for line in p.stdout:
        print(line)
    for line in p.stderr:
        print(line)
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (
        p.returncode,
        num_procs,
    )
