"""
Tests for the Firedrake port of StroemungsRaum's monolithic Navier-Stokes problem. They live here rather than in the
project, because the project's own tests run in its FEniCS environment, and these need Firedrake.
"""

import pytest

# One step of dt = 0.01 from t0 = 3.125e-4 with two RADAU-RIGHT nodes, LU and exactly three iterations, computed with
# `fenics_NSE_2D_Monolithic` (FEniCS 2019.1) on the same mesh: drag and lift, then u_x, u_y and p at each point.
# Kept here so the comparison survives FEniCS.
FENICS_DRAG_LIFT = (0.1451704069871256, -0.00015860186359113043)
FENICS_POINTS = {
    (0.15, 0.3): (0.0049846777788335505, 0.0010062974265752206, 0.7879720048750247),
    (0.5, 0.2): (0.004083777676972536, -4.403916963914531e-06, 0.6411932883458746),
    (1.0, 0.1): (0.004128867874274208, 2.4780838249604534e-06, 0.45319108272710995),
    (2.0, 0.3): (0.004168834116699332, -1.3878135594237236e-06, 0.07560675320216911),
}
# drag and lift at t = 0.0103125 from the FEATFLOW benchmark solution
FEATFLOW_DRAG_LIFT = (1.5689678714e-01, -2.2719989899e-04)


@pytest.mark.firedrake
@pytest.mark.parallel([1, 2])
def test_reproduces_FEniCS():
    """
    The same step as the FEniCS version gives the same drag, lift and point values to round-off, serially and with
    the mesh distributed over two ranks. Like the FEniCS version, it is within the coarse-mesh error of FEATFLOW.
    """
    import firedrake as fd
    import numpy as np
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass
    from pySDC.projects.StroemungsRaum.problem_classes.NavierStokes_2D_monolithic_Firedrake import (
        NavierStokes2DMonolithicFiredrake,
    )

    description = {
        'problem_class': NavierStokes2DMonolithicFiredrake,
        'problem_params': {},
        'sweeper_class': generic_implicit_mass,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 2, 'QI': 'LU'},
        'level_params': {'dt': 0.01, 'restol': -1.0},
        'step_params': {'maxiter': 3},
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    P = controller.MS[0].levels[0].prob
    t0 = 3.125e-4
    uend, _ = controller.run(u0=P.u_exact(t0), t0=t0, Tend=0.01)

    drag_lift = P.drag_lift(uend)
    assert np.allclose(drag_lift, FENICS_DRAG_LIFT, rtol=0, atol=1e-12), drag_lift

    u, p = uend.functionspace.subfunctions
    evaluate = fd.PointEvaluator(P.mesh, [list(x) for x in FENICS_POINTS]).evaluate
    values = np.column_stack([evaluate(u), evaluate(p)])
    assert np.allclose(values, list(FENICS_POINTS.values()), rtol=0, atol=1e-12), values

    # the errors of this coarse mesh, as measured for the FEniCS version
    errors = np.abs(np.array(drag_lift) - FEATFLOW_DRAG_LIFT)
    assert errors[0] < 2e-2 and errors[1] < 1e-4, errors


@pytest.mark.firedrake
def test_boundary_ids():
    """The mesh read from the DOLFIN file has its boundary facets labelled, with the expected lengths."""
    import firedrake as fd
    import numpy as np
    from pySDC.projects.StroemungsRaum.problem_classes.NavierStokes_2D_monolithic_Firedrake import (
        CYLINDER,
        INFLOW,
        OUTFLOW,
        WALLS,
        cylinder_mesh,
    )

    mesh = cylinder_mesh()
    one = fd.Constant(1.0)
    lengths = [fd.assemble(one * fd.ds(i, domain=mesh)) for i in (INFLOW, OUTFLOW, WALLS, CYLINDER)]
    # the cylinder is a polygon through points of the circle of radius 0.05
    assert np.allclose(lengths[:3], [0.41, 0.41, 4.4]), lengths
    assert np.isclose(lengths[3], 2 * np.pi * 0.05, rtol=1e-2), lengths
    assert np.isclose(sum(lengths), fd.assemble(one * fd.ds(domain=mesh))), lengths
