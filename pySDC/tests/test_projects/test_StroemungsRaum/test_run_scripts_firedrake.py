"""
Tests for the Firedrake ports of StroemungsRaum's Taylor-Green and monolithic run scripts and of the heat accuracy
study, mirroring what the FEniCS tests assert about them.
"""

import pytest


@pytest.mark.firedrake
@pytest.mark.parametrize('periodic', [False, True])
def test_taylor_green_benchmark(periodic):
    """Two steps land close to the exact solution; the tolerance is the spatial error of this coarse mesh."""
    from pySDC.projects.StroemungsRaum.run_Navier_Stokes_TaylorGreen_Firedrake import (
        run_postprocessing,
        run_simulation,
        setup,
    )

    description, controller_params, t0 = setup(dt=0.05, periodic=periodic, nelems=16, nu=0.05, num_nodes=2)
    P, _, uend = run_simulation(description, controller_params, t0, Tend=0.1)
    error_u, error_p = run_postprocessing(P, uend, Tend=0.1)
    assert error_u < 1e-3, error_u
    assert error_p < 1e-2, error_p


@pytest.mark.firedrake
def test_monolithic_run_script():
    """One step of the monolithic cylinder flow, within the coarse-mesh tolerances of the FEniCS test against FEATFLOW."""
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.projects.StroemungsRaum.run_Navier_Stokes_equations_monolithic_Firedrake import run_simulation, setup

    description, controller_params, t0 = setup()
    controller_params['logger_level'] = 30
    _, stats, _ = run_simulation(description, controller_params, t0, Tend=0.01)
    drag = get_sorted(stats, type='drag_post_step')[-1][1]
    lift = get_sorted(stats, type='lift_post_step')[-1][1]
    # FEATFLOW at t = 0.0103125
    assert abs(drag - 1.5689678714e-01) < 2e-2, drag
    assert abs(lift + 2.2719989899e-04) < 1e-4, lift


@pytest.mark.firedrake
@pytest.mark.slow
def test_heat_3rd_order(tmp_path):
    """Two RADAU-RIGHT nodes give order 2M-1 = 3; 64 x 64 squares keep the spatial error below the time error."""
    import matplotlib

    from pySDC.projects.StroemungsRaum.run_accuracy.run_accuracy_heat_equation_Firedrake import (
        plot_accuracy,
        run_accuracy,
    )

    matplotlib.use('Agg')
    results, order = run_accuracy(c_nvars=64, num_nodes=2)
    assert abs(order - 3) < 0.1, order
    plot_accuracy(results, 3, path=str(tmp_path))
    assert (tmp_path / 'heat_equation_3rd_order_time_Firedrake.png').is_file()
