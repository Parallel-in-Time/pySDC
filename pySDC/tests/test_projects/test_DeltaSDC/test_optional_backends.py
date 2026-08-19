"""
Optional-backend coverage for the DeltaSDC project.

These live under ``pySDC/tests`` rather than in the project's own test folder because the CI job
that installs FEniCS, PETSc and mpi4py selects tests by marker (``pytest pySDC/tests -m <env>``),
while the project job installs only the project's own environment. Same split as
``pySDC/tests/test_tutorials/test_step_7.py``: the logic lives in runnable project scripts and the
tests here just call them.
"""

import os
import subprocess

import pytest


@pytest.mark.fenics
def test_fenics():
    """Delta-form IMEX on the FEniCS heat equation, against the stock imex_1st_order path."""
    from pySDC.projects.DeltaSDC.run_fenics import main

    main()


@pytest.mark.petsc
def test_petsc():
    """Delta-form Generalized Fisher against the stock path, plus emulated reduced precision."""
    from pySDC.projects.DeltaSDC.run_petsc import main

    main()


def _run_mpi(extra_args):
    """Spawn the node-parallel driver with one rank per collocation node."""
    try:
        import mpi4py

        del mpi4py
    except ImportError:
        raise ImportError('the node-parallel sweeper test needs mpi4py')

    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'

    cmd = ('mpirun -np 3 python pySDC/projects/DeltaSDC/run_mpi.py ' + extra_args).split()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env, cwd='.')
    p.wait()
    for line in p.stdout:
        print(line)
    for line in p.stderr:
        print(line)
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s' % p.returncode


@pytest.mark.mpi4py
def test_mpi_sweeper():
    """One collocation node per rank must reproduce the serial delta form and generic_implicit."""
    _run_mpi('')


@pytest.mark.mpi4py
def test_mpi_sweeper_reduced_precision():
    """The same, with the small quantities stored at reduced precision."""
    _run_mpi('--fp32')
