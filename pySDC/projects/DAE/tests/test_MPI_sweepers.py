import pytest


@pytest.mark.mpi4py
@pytest.mark.parametrize('num_procs', [2, 3])
def testOrder(num_procs):
    r"""
    Test checks if order of accuracy is reached for the MPI sweepers.
    """

    import pySDC.projects.DAE.run.accuracy_check_MPI as acc
    import os
    import subprocess

    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'
    cwd = '.'

    cmd = f"mpirun -np {num_procs} python {acc.__file__}".split()
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


@pytest.mark.mpi4py
@pytest.mark.parametrize("num_nodes", [2])
@pytest.mark.parametrize("residual_type", ['full_abs', 'last_abs', 'full_rel', 'last_rel'])
@pytest.mark.parametrize("semi_implicit", [True, False])
@pytest.mark.parametrize("index_case", [1, 2])
@pytest.mark.parametrize("initial_guess", ['spread', 'zero', 'something_else'])
def testVersions(num_nodes, residual_type, semi_implicit, index_case, initial_guess, launch=True):
    r"""
    Make a test if the result matches between the MPI and non-MPI versions of a sweeper.
    Tests solution at the right end point and the residual.

    Parameters
    ----------
    num_nodes : int
        Number of collocation nodes to use.
    residual_type : str
        Type of residual computation.
    semi_implicit : bool
        If True, semi-implicit sweeper is used.
    index_case : int
        Case of DAE index. Choose either between :math:`1` or :math:`2`.
    initial_guess : str
        Type of initial guess for simulation.
    launch : bool
        If yes, it will launch `mpirun` with the required number of processes
    """

    if launch:
        import os
        import subprocess

        # Set python path once
        my_env = os.environ.copy()
        my_env['PYTHONPATH'] = '../../..:.'
        my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'
        cmd = f"mpirun -np {num_nodes} python {__file__} --testVersions {num_nodes} {residual_type} {semi_implicit} {index_case} {initial_guess}".split()

        p = subprocess.Popen(cmd, env=my_env, cwd=".")

        p.wait()
        assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (
            p.returncode,
            num_nodes,
        )
    else:
        import numpy as np
        from pySDC.projects.DAE.run.accuracy_check_MPI import run
        from pySDC.core.errors import ParameterError

        semi_implicit = False if semi_implicit == 'False' else True

        dt = 0.1

        if initial_guess == 'something_else':
            with pytest.raises(ParameterError):
                _, _, _ = run(
                    dt=dt,
                    num_nodes=int(num_nodes),
                    use_MPI=True,
                    semi_implicit=semi_implicit,
                    residual_type=residual_type,
                    index_case=int(index_case),
                    initial_guess=initial_guess,
                )

        else:
            MPI_uend, MPI_residual, _ = run(
                dt=dt,
                num_nodes=int(num_nodes),
                use_MPI=True,
                semi_implicit=semi_implicit,
                residual_type=residual_type,
                index_case=int(index_case),
                initial_guess=initial_guess,
            )

            nonMPI_uend, nonMPI_residual, _ = run(
                dt=dt,
                num_nodes=int(num_nodes),
                use_MPI=False,
                semi_implicit=semi_implicit,
                residual_type=residual_type,
                index_case=int(index_case),
                initial_guess=initial_guess,
            )

            assert np.allclose(MPI_uend, nonMPI_uend, atol=1e-14), 'Got different solutions at end point!'
            assert np.allclose(MPI_residual, nonMPI_residual, atol=1e-14), 'Got different residuals!'


if __name__ == '__main__':
    import sys

    if '--testVersions' in sys.argv:
        testVersions(sys.argv[-5], sys.argv[-4], sys.argv[-3], sys.argv[-2], sys.argv[-1], launch=False)
