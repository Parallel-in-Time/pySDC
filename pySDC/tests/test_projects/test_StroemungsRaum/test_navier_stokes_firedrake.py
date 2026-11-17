"""
Tests for the Firedrake port of StroemungsRaum's monolithic Navier-Stokes problem. They live here rather than in the
project, because the project's own tests run in its FEniCS environment, and these need Firedrake.
"""

import pytest

# One step of dt = 0.01 from t0 = 3.125e-4 with two RADAU-RIGHT nodes, LU and exactly three iterations: drag and lift,
# then u_x, u_y and p at each point, recorded with this class (Firedrake 2026.10) as a check against unintended changes.
REFERENCE_DRAG_LIFT = (0.1451704069871256, -0.00015860186359113043)
REFERENCE_POINTS = {
    (0.15, 0.3): (0.0049846777788335505, 0.0010062974265752206, 0.7879720048750247),
    (0.5, 0.2): (0.004083777676972536, -4.403916963914531e-06, 0.6411932883458746),
    (1.0, 0.1): (0.004128867874274208, 2.4780838249604534e-06, 0.45319108272710995),
    (2.0, 0.3): (0.004168834116699332, -1.3878135594237236e-06, 0.07560675320216911),
}
# drag and lift at t = 0.0103125 from the FEATFLOW benchmark solution
FEATFLOW_DRAG_LIFT = (1.5689678714e-01, -2.2719989899e-04)


@pytest.mark.firedrake
@pytest.mark.parallel([1, 2])
def test_reference_values():
    """
    One step gives the recorded drag, lift and point values to round-off, serially and with the mesh distributed over
    two ranks, and is within the coarse-mesh error of FEATFLOW.
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
    assert np.allclose(drag_lift, REFERENCE_DRAG_LIFT, rtol=0, atol=1e-12), drag_lift

    u, p = uend.functionspace.subfunctions
    evaluate = fd.PointEvaluator(P.mesh, [list(x) for x in REFERENCE_POINTS]).evaluate
    values = np.column_stack([evaluate(u), evaluate(p)])
    assert np.allclose(values, list(REFERENCE_POINTS.values()), rtol=0, atol=1e-12), values

    # the errors of this coarse mesh against FEATFLOW
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


# The boundary treatments in x: time-dependent Dirichlet data, the same data in differentiated form, or periodic.
TAYLOR_GREEN_CASES = {
    'dirichlet': dict(periodic=False),
    'differentiated': dict(periodic=False, differentiated_bc=True),
    'periodic': dict(periodic=True),
}
# u_x, u_y and p of the Taylor-Green problem after two steps of dt = 0.05 (16 x 16 squares, nu = 0.05, two
# RADAU-RIGHT nodes, LU, exactly three iterations, Newton tolerances 1e-13), recorded with this class (Firedrake
# 2026.10) as a check against unintended changes.
TAYLOR_GREEN_POINTS = [[0.1, 0.2], [-0.3, -0.1], [0.45, 0.3], [-0.2, 0.4]]
TAYLOR_GREEN_REFERENCE = {
    'dirichlet': [
        [1.000205728005674, -0.4411661474845772, 1.0836654413300941],
        [0.8825937492201837, 0.49307129318579124, 1.0316782773977022],
        [0.7411286526349095, 0.13678101793746505, 0.9811090637062253],
        [1.1876223478752206, 0.01923599118929458, 0.9713018191938174],
    ],
    'differentiated': [
        [1.0002255713288684, -0.4411855152950595, 1.083564557964147],
        [0.8825130137574685, 0.4929627990452926, 1.0318914731488762],
        [0.7412643144208172, 0.13660641089482248, 0.9812586559176729],
        [1.187581060345629, 0.01914633454243825, 0.9710083365238055],
    ],
    'periodic': [
        [1.0002569069703042, -0.44086132180666576, 1.0844231070331143],
        [0.8834034160760434, 0.4928050573665194, 1.0321255333274153],
        [0.7410536287474639, 0.13728516511170485, 0.9811513691777825],
        [1.1878002932967076, 0.01979576470764152, 0.972459948153029],
    ],
}


def taylor_green(dt, Tend, case, num_nodes, maxiter, restol, nelems, nu):
    """Run the Taylor-Green problem with SDC for one of TAYLOR_GREEN_CASES and return the problem and the end point."""
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass
    from pySDC.projects.StroemungsRaum.sweepers.generic_implicit_mass_diffbc import generic_implicit_mass_diffbc
    from pySDC.projects.StroemungsRaum.problem_classes.NavierStokes_2D_TaylorGreen_monolithic_Firedrake import (
        NavierStokes2DTaylorGreenFiredrake,
    )

    description = {
        'problem_class': NavierStokes2DTaylorGreenFiredrake,
        'problem_params': {
            'nelems': nelems,
            'nu': nu,
            **TAYLOR_GREEN_CASES[case],
            'solver_parameters': {'snes_atol': 1e-13, 'snes_rtol': 1e-13},
        },
        'sweeper_class': generic_implicit_mass_diffbc if case == 'differentiated' else generic_implicit_mass,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': num_nodes, 'QI': 'LU'},
        'level_params': {'dt': dt, 'restol': restol},
        'step_params': {'maxiter': maxiter},
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    P = controller.MS[0].levels[0].prob
    uend, _ = controller.run(u0=P.u_exact(0.0), t0=0.0, Tend=Tend)
    return P, uend


@pytest.mark.firedrake
@pytest.mark.parametrize('case', list(TAYLOR_GREEN_CASES))
def test_taylor_green_reference_values(case):
    """Two steps give the recorded values to round-off, for each boundary treatment in x."""
    import firedrake as fd
    import numpy as np

    P, uend = taylor_green(0.05, 0.1, case, num_nodes=2, maxiter=3, restol=-1.0, nelems=16, nu=0.05)
    u, p = uend.functionspace.subfunctions
    evaluate = fd.PointEvaluator(P.mesh, TAYLOR_GREEN_POINTS).evaluate
    values = np.column_stack([evaluate(u), evaluate(p)])
    assert np.allclose(values, TAYLOR_GREEN_REFERENCE[case], rtol=0, atol=1e-12), values


@pytest.mark.firedrake
def test_taylor_green_order_reduction():
    """
    With four RADAU-RIGHT nodes, the same exact solution reaches the design order 7 with periodic boundaries in x,
    while time-dependent Dirichlet data there pulls the pressure down towards the stiff order 5, and the same data in
    differentiated form recovers most of the loss. Consecutive step sizes are compared with each other, which cancels
    the spatial error. This is the order study of the project's run script, on a coarser mesh.
    """
    from pySDC.projects.StroemungsRaum.run_Navier_Stokes_TaylorGreen_Firedrake import observed_order, order_study

    dts, Tend, num_nodes = [0.2, 0.1, 0.05, 0.025], 0.2, 4
    orders, finest = {}, {}
    for case, kwargs in TAYLOR_GREEN_CASES.items():
        dts_out, errors_u, errors_p = order_study(dts, Tend, nelems=16, num_nodes=num_nodes, **kwargs)
        orders[case] = [observed_order(dts_out, errors)[-1] for errors in (errors_u, errors_p)]
        finest[case] = [errors_u[-1], errors_p[-1]]

    design, stiff = 2 * num_nodes - 1, num_nodes + 1
    assert orders['periodic'][0] > design - 1.0, orders
    assert orders['dirichlet'][1] < (design + stiff) / 2, orders
    assert orders['periodic'][1] - orders['dirichlet'][1] > 1.0, orders
    assert finest['dirichlet'][1] / finest['periodic'][1] > 10.0, finest
    # the differentiated data recover most of it
    assert orders['differentiated'][1] - orders['dirichlet'][1] > 1.0, orders
    assert finest['dirichlet'][1] / finest['differentiated'][1] > 5.0, finest


# The projection scheme after two steps of dt = 1/1600 from t0 = 3.125e-4 (two RADAU-RIGHT nodes, LU implicit, Picard
# explicit, exactly three iterations, `imex_1st_order_mass_NSE`): drag and lift, then u_x, u_y and p at the points of
# REFERENCE_POINTS, recorded with this class as a check against unintended changes.
PROJECTION_DRAG_LIFT = (0.1412337727444787, -0.00027963313489825364)
PROJECTION_POINTS = [
    [0.0007391793277933225, 0.0001430702172714647, 0.8404115351086197],
    [0.0006138307177725508, -8.080310231710478e-07, 0.6859610805277692],
    [0.0006170498773377232, 2.742331940577588e-07, 0.48537034723473604],
    [0.000625531433988877, -1.72714695699517e-07, 0.08102075727895316],
]


def projection():
    """Run the two projection steps of PROJECTION_DRAG_LIFT and return the problem and the end point."""
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.projects.StroemungsRaum.problem_classes.NavierStokes_2D_projection_Firedrake import (
        NavierStokes2DProjectionFiredrake,
    )
    from pySDC.projects.StroemungsRaum.sweepers.imex_1st_order_mass_NSE import imex_1st_order_mass_NSE

    dt, t0 = 1 / 1600, 3.125e-4
    description = {
        'problem_class': NavierStokes2DProjectionFiredrake,
        'problem_params': {},
        'sweeper_class': imex_1st_order_mass_NSE,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 2, 'QI': 'LU', 'QE': 'PIC'},
        'level_params': {'dt': dt, 'restol': -1.0},
        'step_params': {'maxiter': 3},
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    P = controller.MS[0].levels[0].prob
    uend, _ = controller.run(u0=P.u_exact(t0), t0=t0, Tend=t0 + 2 * dt)
    return P, uend


@pytest.mark.firedrake
def test_projection_reference_values():
    """The same steps of the projection scheme give the recorded values, to round-off."""
    import firedrake as fd
    import numpy as np

    P, uend = projection()
    assert np.allclose(P.drag_lift(uend), PROJECTION_DRAG_LIFT, rtol=0, atol=1e-12)

    evaluate = fd.PointEvaluator(P.mesh, [list(x) for x in REFERENCE_POINTS]).evaluate
    values = np.column_stack([evaluate(uend.functionspace), evaluate(P.pn)])
    assert np.allclose(values, PROJECTION_POINTS, rtol=0, atol=1e-12), values
