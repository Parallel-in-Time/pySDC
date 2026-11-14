"""
Tests for the Firedrake ports of StroemungsRaum's hooks, run scripts and plots.
"""

import pytest


@pytest.mark.firedrake
def test_lift_drag_hook():
    """
    The drag and lift the hook records at the last step are those of the FEniCS hook (same iterations, so the same
    values to ten digits), within the tolerances of the FEniCS test against FEATFLOW.
    """
    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.projects.StroemungsRaum.run_Navier_Stokes_equations_Firedrake import run_simulation, setup

    description, controller_params, t0 = setup()
    controller_params['logger_level'] = 30
    P, stats, uend = run_simulation(description, controller_params, t0, Tend=0.005)
    drag = get_sorted(stats, type='drag_post_step', sortby='time')
    lift = get_sorted(stats, type='lift_post_step', sortby='time')

    assert len(drag) == len(lift) == 8
    assert np.allclose([drag[-1][1], lift[-1][1]], P.drag_lift(uend), rtol=1e-12, atol=0)
    # the last values of the FEniCS hook, LogLiftDrag of hooks_NSE_IMEX_FEniCS, with the same setup
    assert np.allclose([drag[-1][1], lift[-1][1]], [1.4635256617e-01, -1.9517335230e-04], rtol=1e-9, atol=0)
    # FEATFLOW at t = 0.0046875
    assert abs(drag[-1][1] - 1.4903278329e-01) < 5e-3 and abs(lift[-1][1] + 2.3003337267e-04) < 5e-5


@pytest.mark.firedrake
def test_write_solutions_hook_and_plots(tmp_path, monkeypatch):
    """The solutions the hook writes give back the drag and lift of the run, and the plots are drawn from them."""
    import firedrake as fd
    import matplotlib
    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.projects.StroemungsRaum.hooks.hooks_NSE_IMEX_Firedrake import LogWriteSolutions
    from pySDC.projects.StroemungsRaum.plotting.plots_Navier_Stokes_projection_Firedrake import plot_solutions
    from pySDC.projects.StroemungsRaum.problem_classes.NavierStokes_2D_monolithic_Firedrake import drag_lift
    from pySDC.projects.StroemungsRaum.run_Navier_Stokes_equations_Firedrake import (
        run_postprocessing,
        run_simulation,
        setup,
    )

    matplotlib.use('Agg')
    monkeypatch.setattr(LogWriteSolutions, 'path', str(tmp_path))
    description, controller_params, t0 = setup()
    controller_params['logger_level'] = 30
    controller_params['hook_class'].append(LogWriteSolutions)
    P, stats, _ = run_simulation(description, controller_params, t0, Tend=t0 + 2 * description['level_params']['dt'])
    run_postprocessing(description, stats, path=str(tmp_path))

    with fd.CheckpointFile(f'{tmp_path}/Cylinder_solutions.h5', 'r') as file:
        mesh = file.load_mesh()
        un, pn = file.load_function(mesh, 'un', idx=1), file.load_function(mesh, 'pn', idx=1)
        stored = drag_lift(un, pn, P.nu, mesh)
    logged = [get_sorted(stats, type=f'{kind}_post_step', sortby='time')[-1][1] for kind in ('drag', 'lift')]
    assert np.allclose(stored, logged, rtol=1e-12, atol=0)
    assert len(open(f'{tmp_path}/Lift_Drag_Coefficients.txt').readlines()) == 2

    plot_solutions(path=str(tmp_path), animate=False)
    assert (tmp_path / 'Cylinder_Firedrake.png').is_file()


@pytest.mark.firedrake
@pytest.mark.parametrize('example', ['heat_equation', 'convection_diffusion_equation'])
def test_run_postprocessing_and_plots(example, tmp_path):
    """The run script stores every step's numerical and exact solution, and the plots are drawn from them."""
    import importlib

    import firedrake as fd
    import matplotlib
    import numpy as np

    matplotlib.use('Agg')
    run = importlib.import_module(f'pySDC.projects.StroemungsRaum.run_{example}_Firedrake')
    plots = importlib.import_module(f'pySDC.projects.StroemungsRaum.plotting.plots_{example}_Firedrake')

    description, controller_params, t0 = run.setup()
    controller_params['logger_level'] = 30
    Tend = t0 + 2 * description['level_params']['dt']
    P, stats, rel_err = run.run_simulation(description, controller_params, t0, Tend)
    run.run_postprocessing(description, P, stats, Tend, path=str(tmp_path))

    name = (
        'heat_equation_Firedrake_Temperature'
        if example == 'heat_equation'
        else 'convection_diffusion_Firedrake_solutions'
    )
    with fd.CheckpointFile(f'{tmp_path}/{name}.h5', 'r') as file:
        mesh = file.load_mesh()
        un, ux = file.load_function(mesh, 'un', idx=1), file.load_function(mesh, 'ux', idx=1)
    exact = P.u_exact(Tend)
    assert np.isclose(fd.errornorm(ux, un) / fd.norm(ux), rel_err, rtol=1e-10)
    assert np.allclose(ux.dat.data_ro, exact.dat.data_ro)

    plots.plot_solutions(path=str(tmp_path), animate=False)
    prefix = name.rsplit('_', 1)[0]
    assert (tmp_path / f'{prefix}_Results.png').is_file() and (tmp_path / f'{prefix}_Contours.png').is_file()
