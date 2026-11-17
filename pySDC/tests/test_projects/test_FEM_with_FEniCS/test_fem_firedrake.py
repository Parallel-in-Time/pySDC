"""
Tests for the Firedrake ports of the FEM_with_FEniCS problems. They live here rather than in the project, because the
project's own tests run in its FEniCS environment, and these need Firedrake.
"""

import pytest

TOL = 1e-13
SOLVER = {'snes_atol': TOL, 'snes_rtol': TOL}

# Each case runs two steps with three RADAU-RIGHT nodes, LU and exactly three iterations. The reference values at the
# points were computed with these classes (Firedrake 2026.10, default Lagrange nodes, Newton tolerances 1e-13 where
# there is a Newton loop), as a check against unintended changes: (class, parameters, dt, IMEX, points, values).
CASES = {
    'burgers': (
        'Burgers_1D_Firedrake.Burgers1DFiredrake',
        dict(n=32, order=4, nu=0.02, solver_parameters=SOLVER),
        0.02,
        False,
        [0.1, 0.37, 0.61, 0.83],
        [[0.8879057355800254], [-0.7848223212380506], [-0.6409229179606641], [0.3848063203202955]],
    ),
    'heat_dg': (
        'DG_1D_Firedrake.Heat1DDGFiredrake',
        dict(n=16, order=4, nu=0.1, c=1.0),
        0.1,
        True,
        [0.1, 0.37, 0.61, 0.83],
        [[1.3028572476996685], [1.899460711092855], [1.922125871940941], [1.4988945137952487]],
    ),
    'burgers_dg': (
        'DG_1D_Firedrake.Burgers1DDGFiredrake',
        dict(n=32, order=4, nu=0.02, solver_parameters=SOLVER),
        0.02,
        False,
        [0.1, 0.37, 0.61, 0.83],
        [[0.8879057355826151], [-0.7848223204487138], [-0.6409229179680823], [0.3848063203080977]],
    ),
    'grayscott_dg': (
        'DG_1D_Firedrake.GrayScott1DDGFiredrake',
        dict(n=32, order=4, solver_parameters=SOLVER),
        1.0,
        False,
        [40.1, 45.0, 49.0, 52.3],
        [
            [0.9891725727347049, 0.0015757858773579687],
            [0.8427400292048042, 0.06898154472285171],
            [0.5995921644855744, 0.265678672479725],
            [0.6621086981216051, 0.20931950349326836],
        ],
    ),
}


@pytest.mark.firedrake
@pytest.mark.parametrize('case', list(CASES))
def test_reference_values(case):
    """The same steps give the recorded values, to round-off."""
    import importlib

    import firedrake as fd
    import numpy as np
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass
    from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass

    name, params, dt, imex, points, expect = CASES[case]
    module, class_name = name.split('.')
    problem_class = getattr(
        importlib.import_module(f'pySDC.projects.FEM_with_FEniCS.problem_classes.{module}'), class_name
    )

    description = {
        'problem_class': problem_class,
        'problem_params': params,
        'sweeper_class': imex_1st_order_mass if imex else generic_implicit_mass,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'LU'},
        'level_params': {'dt': dt, 'restol': -1.0},
        'step_params': {'maxiter': 3},
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    P = controller.MS[0].levels[0].prob
    uend, _ = controller.run(u0=P.u_exact(0.0), t0=0.0, Tend=2 * dt)

    parts = uend.functionspace.subfunctions
    evaluate = fd.PointEvaluator(parts[0].function_space().mesh(), [[x] for x in points]).evaluate
    values = np.column_stack([evaluate(part) for part in parts])
    assert np.allclose(values, expect, rtol=0, atol=1e-11), np.abs(values - np.array(expect)).max()


@pytest.mark.firedrake
def test_initial_value_only():
    """Burgers and Gray-Scott have no exact solution, only an initial value."""
    from pySDC.projects.FEM_with_FEniCS.problem_classes.Burgers_1D_Firedrake import Burgers1DFiredrake
    from pySDC.projects.FEM_with_FEniCS.problem_classes.DG_1D_Firedrake import (
        Burgers1DDGFiredrake,
        GrayScott1DDGFiredrake,
    )

    for problem_class in [Burgers1DFiredrake, Burgers1DDGFiredrake, GrayScott1DDGFiredrake]:
        P = problem_class(n=4, order=1)
        P.u_exact(0)
        with pytest.raises(AssertionError):
            P.u_exact(0.5)


@pytest.mark.firedrake
def test_study_heat_h():
    """
    The study's heat example, CG4 with h-coarsening, gives the speed-ups in the README (1.24x on two levels, 1.57x on
    three), and MLSDC converges to the SDC solution.
    """
    import numpy as np
    from pySDC.projects.FEM_with_FEniCS.run_examples_firedrake import compare_mlsdc, l2_diff

    results = compare_mlsdc('heat', out=lambda *args: None)
    speedups = [results[1]['work'] / results[n]['work'] for n in (2, 3)]
    assert np.allclose(speedups, [1.24, 1.57], atol=0.01), speedups
    for n in (2, 3):
        assert l2_diff(results[n]['uend'], results[1]['uend']) < 1e-8
