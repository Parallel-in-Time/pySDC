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


# u_x, u_y and p of the Taylor-Green problem after two steps of dt = 0.05 (16 x 16 squares, nu = 0.05, two
# RADAU-RIGHT nodes, LU, exactly three iterations), computed with `fenics_NSE_2D_TaylorGreen` and its Newton
# tolerances tightened to 1e-13 / 1e-15, for time-dependent Dirichlet (False) and periodic (True) boundaries in x.
TAYLOR_GREEN_POINTS = [[0.1, 0.2], [-0.3, -0.1], [0.45, 0.3], [-0.2, 0.4]]
TAYLOR_GREEN_FENICS = {
    False: [
        [0.999726491161949, -0.44125662925840736, 1.0832928076702906],
        [0.8829651353597946, 0.49320504968302314, 1.0316227614608828],
        [0.7415568521416037, 0.13663961216880038, 0.9797246235242725],
        [1.1884337485737229, 0.019251220181781426, 0.9719320069725023],
    ],
    True: [
        [0.9996888211906914, -0.44100097173070424, 1.084260314424162],
        [0.8837206570498065, 0.49289575383441103, 1.0319411694124747],
        [0.740913013555852, 0.13710285013486995, 0.980577482854738],
        [1.1886075690931954, 0.019843126697488833, 0.9733450742555501],
    ],
}


def taylor_green(dt, Tend, periodic, num_nodes, maxiter, restol, nelems, nu):
    """Run the Taylor-Green problem with SDC and return the problem and the end point."""
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass
    from pySDC.projects.StroemungsRaum.problem_classes.NavierStokes_2D_TaylorGreen_monolithic_Firedrake import (
        NavierStokes2DTaylorGreenFiredrake,
    )

    description = {
        'problem_class': NavierStokes2DTaylorGreenFiredrake,
        'problem_params': {
            'nelems': nelems,
            'nu': nu,
            'periodic': periodic,
            'solver_parameters': {'snes_atol': 1e-13, 'snes_rtol': 1e-13},
        },
        'sweeper_class': generic_implicit_mass,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': num_nodes, 'QI': 'LU'},
        'level_params': {'dt': dt, 'restol': restol},
        'step_params': {'maxiter': maxiter},
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    P = controller.MS[0].levels[0].prob
    uend, _ = controller.run(u0=P.u_exact(0.0), t0=0.0, Tend=Tend)
    return P, uend


def relative_differences(u, ref):
    """Relative L2 differences in velocity and pressure of two solutions of identically built problems."""
    import firedrake as fd

    difference = fd.Function(ref.function_space())
    for d, a, b in zip(difference.dat.data_wo, u.dat.data_ro, ref.dat.data_ro, strict=True):
        d[:] = a - b
    return [
        fd.norm(d) / fd.norm(r) for d, r in zip(difference.subfunctions, ref.functionspace.subfunctions, strict=True)
    ]


@pytest.mark.firedrake
@pytest.mark.parametrize('periodic', [False, True])
def test_taylor_green_reproduces_FEniCS(periodic):
    """
    The same steps as the FEniCS version give the same solution to round-off, with time-dependent Dirichlet and with
    periodic boundaries in x. This needs the forcing interpolated with equispaced nodes, as DOLFIN does: with
    Firedrake's default nodes for degree 4 the two differ by 3e-8.
    """
    import firedrake as fd
    import numpy as np

    P, uend = taylor_green(0.05, 0.1, periodic, num_nodes=2, maxiter=3, restol=-1.0, nelems=16, nu=0.05)
    u, p = uend.functionspace.subfunctions
    evaluate = fd.PointEvaluator(P.mesh, TAYLOR_GREEN_POINTS).evaluate
    values = np.column_stack([evaluate(u), evaluate(p)])
    assert np.allclose(values, TAYLOR_GREEN_FENICS[periodic], rtol=0, atol=1e-12), values


@pytest.mark.firedrake
def test_taylor_green_order_reduction():
    """
    The finding of the FEniCS version: with four RADAU-RIGHT nodes, the same exact solution reaches the design order
    7 with periodic boundaries in x, while time-dependent Dirichlet data there pulls the pressure down towards the
    stiff order 5. Consecutive step sizes are compared with each other, which cancels the spatial error.
    """
    import numpy as np

    dts, Tend, num_nodes = [0.2, 0.1, 0.05, 0.025], 0.2, 4
    orders, finest = {}, {}
    for periodic in [True, False]:
        solutions = [taylor_green(dt, Tend, periodic, num_nodes, 40, 1e-11, 16, 0.1)[1] for dt in dts]
        errors = np.array([relative_differences(u, ref) for u, ref in zip(solutions[:-1], solutions[1:], strict=True)])
        orders[periodic] = np.log2(errors[:-1] / errors[1:])[-1]
        finest[periodic] = errors[-1]

    design, stiff = 2 * num_nodes - 1, num_nodes + 1
    assert orders[True][0] > design - 1.0, orders
    assert orders[False][1] < (design + stiff) / 2, orders
    assert orders[True][1] - orders[False][1] > 1.0, orders
    assert finest[False][1] / finest[True][1] > 10.0, finest


# The projection scheme after two steps of dt = 1/1600 from t0 = 3.125e-4 (two RADAU-RIGHT nodes, LU implicit, Picard
# explicit, exactly three iterations), computed with `fenics_NSE_2D_mass` and `imex_1st_order_mass_NSE`: drag and
# lift, then u_x, u_y and p at the points of FENICS_POINTS.
PROJECTION_FENICS_DRAG_LIFT = (0.1412337727444787, -0.00027963313489825364)
PROJECTION_FENICS_POINTS = [
    [0.0007391793277933225, 0.0001430702172714647, 0.8404115351086197],
    [0.0006138307177725508, -8.080310231710478e-07, 0.6859610805277692],
    [0.0006170498773377232, 2.742331940577588e-07, 0.48537034723473604],
    [0.000625531433988877, -1.72714695699517e-07, 0.08102075727895316],
]
# drag and lift at t = 0.005 from the FEATFLOW benchmark solution
PROJECTION_FEATFLOW_DRAG_LIFT = (1.4903278329e-01, -2.3003337267e-04)


def projection(level_params, maxiter, nsteps=None, Tend=None):
    """Run the projection scheme from t0 = 3.125e-4 and return the problem, the end point and the iterations."""
    from pySDC.helpers.stats_helper import get_sorted
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
        'level_params': {'dt': dt, **level_params},
        'step_params': {'maxiter': maxiter},
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    P = controller.MS[0].levels[0].prob
    uend, stats = controller.run(u0=P.u_exact(t0), t0=t0, Tend=t0 + nsteps * dt if Tend is None else Tend)
    return P, uend, [me[1] for me in get_sorted(stats, type='niter')]


@pytest.mark.firedrake
def test_projection_reproduces_FEniCS():
    """The same steps of the projection scheme as the FEniCS version give the same solution to round-off."""
    import firedrake as fd
    import numpy as np

    P, uend, _ = projection({'restol': -1.0}, maxiter=3, nsteps=2)
    assert np.allclose(P.drag_lift(uend), PROJECTION_FENICS_DRAG_LIFT, rtol=0, atol=1e-12)

    evaluate = fd.PointEvaluator(P.mesh, [list(x) for x in FENICS_POINTS]).evaluate
    values = np.column_stack([evaluate(uend.functionspace), evaluate(P.pn)])
    assert np.allclose(values, PROJECTION_FENICS_POINTS, rtol=0, atol=1e-12), values


@pytest.mark.firedrake
def test_projection_benchmark():
    """
    The project's setup, stopping on the increment (`e_tol`), takes the same iterations as the FEniCS version, lands on
    the same drag and lift, and is within the tolerances of the FEniCS test against FEATFLOW.
    """
    import numpy as np

    P, uend, iterations = projection({'e_tol': 1e-9}, maxiter=10, Tend=0.005)
    drag_lift = P.drag_lift(uend)

    assert iterations == [4] * 7 + [3], iterations
    assert np.allclose(drag_lift, (1.4635256617e-01, -1.9517335230e-04), rtol=1e-9, atol=0), drag_lift
    errors = np.abs(np.array(drag_lift) - PROJECTION_FEATFLOW_DRAG_LIFT)
    assert errors[0] < 5e-3 and errors[1] < 5e-5, errors
