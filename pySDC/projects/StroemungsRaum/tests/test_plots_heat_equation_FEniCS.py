import pytest
import subprocess
import os
import warnings


@pytest.mark.fenics
@pytest.mark.mpi4py
@pytest.mark.parametrize('mass', [True])
def test_plot(mass):
    from pySDC.projects.StroemungsRaum.run_heat_equation_FEniCS import run_simulation

    run_simulation(mass=mass)

    assert os.path.isfile('data/heat_equation_FEniCS_parameters.json'), 'ERROR: parameters.json does not exist'
    assert os.path.isfile('data/heat_equation_FEniCS_Temperature.xdmf'), 'ERROR: Temperature.xdmf does not exist'
    assert os.path.isfile('data/heat_equation_FEniCS_Temperature.h5'), 'ERROR: Temperature.h5 does not exist'

    from pySDC.projects.StroemungsRaum.plotting.plots_heat_equation_FEniCS import plot_solutions

    plot_solutions()

    assert os.path.isfile('data/heat_equation_FEniCS_Contours.png'), 'ERROR: contours plots has not been created'
    assert os.path.isfile(
        'data/heat_equation_FEniCS_Results.png'
    ), 'ERROR: 3D and cross-sections plots has not been created'
