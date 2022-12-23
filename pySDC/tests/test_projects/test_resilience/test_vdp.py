import pytest
import os
import subprocess


@pytest.mark.mpi4py
def test_main():
    import pySDC.projects.Resilience.vdp as vdp

    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

    # set list of number of parallel steps (even)
    num_procs_list = [1, 2, 5, 8]

    # run code with different number of MPI processes
    for num_procs in num_procs_list:
        cmd = f"mpirun -np {num_procs} python {vdp.__file__}".split()

        p = subprocess.Popen(cmd, env=my_env, cwd=".")

        p.wait()
        assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (
            p.returncode,
            num_procs,
        )


if __name__ == "__main__":
    test_main()
