"""
Optional-backend coverage for the DeltaSDC project.

These live under ``pySDC/tests`` rather than in the project's own test folder because the CI job
that installs FEniCS, PETSc and mpi4py selects tests by marker (``pytest pySDC/tests -m <env>``),
while the project job installs only the project's own environment. Same split as
``pySDC/tests/test_tutorials/test_step_7.py``: the logic lives in runnable project scripts and the
tests here just call them.
"""

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


@pytest.mark.mpi4py
@pytest.mark.parallel(3)
@pytest.mark.parametrize('precision', [None, 'float32'])
def test_mpi_sweeper(precision):
    """One collocation node per rank must reproduce the serial delta form, also with the corrections stored at fp32."""
    import numpy as np

    from pySDC.projects.DeltaSDC.run_mpi import main

    main(None if precision is None else np.dtype(precision))
