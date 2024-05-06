import pytest


@pytest.mark.mpi4py
@pytest.mark.parametrize('num_procs', [1, 2, 5, 8])
@pytest.mark.parametrize('test_name', ['mpi_vs_nonMPI', 'check_step_size_limiter'])
def test_stuff(num_procs, test_name):
    import pySDC.projects.Resilience.vdp as vdp
    import os
    import subprocess

    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

    # run code with different number of MPI processes
    cmd = f"mpirun -np {num_procs} python {vdp.__file__} {test_name}".split()

    p = subprocess.Popen(cmd, env=my_env, cwd=".")

    p.wait()
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (
        p.returncode,
        num_procs,
    )


@pytest.mark.mpi4py
def test_adaptivity_with_avoid_restarts():
    test_stuff(1, 'adaptivity_with_avoid_restarts')


if __name__ == "__main__":
    test_stuff(8, '')
