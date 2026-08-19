import os
import shutil
import subprocess

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
DRIVER = os.path.join('pySDC', 'projects', 'DeltaSDC', 'tests', 'mpi_driver.py')


def _run_driver(extra_args=()):
    launcher = shutil.which('mpirun') or shutil.which('mpiexec')
    if launcher is None:
        pytest.skip('mpirun/mpiexec is required for the node-parallel sweeper test.')

    env = dict(os.environ)
    env['PYTHONPATH'] = REPO_ROOT + os.pathsep + env.get('PYTHONPATH', '')

    command = [launcher, '-np', '3', 'python', DRIVER, *extra_args]
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    return completed


@pytest.mark.mpi4py
def test_node_parallel_matches_serial():
    """One collocation node per rank must reproduce the serial delta form and generic_implicit."""
    completed = _run_driver()
    assert completed.returncode == 0, f'driver failed:\n{completed.stdout}\n{completed.stderr}'
    assert 'OK' in completed.stdout, completed.stdout
    assert 'MISMATCH' not in completed.stdout, completed.stdout


@pytest.mark.mpi4py
def test_node_parallel_with_reduced_precision_corrections():
    """The same, with the small quantities stored at reduced precision."""
    completed = _run_driver(('--fp32',))
    assert completed.returncode == 0, f'driver failed:\n{completed.stdout}\n{completed.stderr}'
    assert 'OK' in completed.stdout, completed.stdout
    assert 'MISMATCH' not in completed.stdout, completed.stdout
