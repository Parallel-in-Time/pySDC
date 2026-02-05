import pytest
import subprocess
import os
import warnings


@pytest.mark.fenics
@pytest.mark.mpi4py
@pytest.mark.parametrize('mass', [True, False])
def test_problem_class(mass):
    from pySDC.projects.FluidFlow.run_heat_equation_FEniCS import run_simulation

    run_simulation(mass=mass)
