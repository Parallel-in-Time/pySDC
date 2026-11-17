"""
Tests for StroemungsRaum's 2D heat and convection-diffusion problems.
"""

import pytest

# Two steps from t = 0 with two RADAU-RIGHT nodes, the default preconditioners and exactly three iterations (heat
# dt = 0.2, convection-diffusion dt = 0.005, default parameters), as u at the points, computed with these classes
# (Firedrake 2026.10) as a check against unintended changes.
REFERENCE = {
    'heat': {
        (0.1, 0.2): 0.1673156273643707,
        (0.5, 0.5): 0.9211559411114335,
        (0.73, 0.31): 0.5714870760711991,
        (0.9, 0.85): 0.12922990958517164,
    },
    'convdiff': {
        (-0.25, 0.0): 0.8330320402839247,
        (-0.2, 0.05): 0.10569585230600223,
        (-0.3, -0.03): 0.31452031127144664,
        (-0.27, 0.02): 0.547796940948503,
    },
}
DT = {'heat': 0.2, 'convdiff': 0.005}


def run(name, dt, Tend, maxiter, restol):
    """Run a problem with its project setup; return the problem and the end point."""
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
    from pySDC.projects.StroemungsRaum.problem_classes.ConvectionDiffusion_2D_Firedrake import ConvDiff2DFiredrake
    from pySDC.projects.StroemungsRaum.problem_classes.HeatEquation_2D_Firedrake import Heat2DFiredrake

    description = {
        'problem_class': {'heat': Heat2DFiredrake, 'convdiff': ConvDiff2DFiredrake}[name],
        'problem_params': {},
        'sweeper_class': imex_1st_order_mass,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 2},
        'level_params': {'dt': dt, 'restol': restol},
        'step_params': {'maxiter': maxiter},
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    P = controller.MS[0].levels[0].prob
    uend, _ = controller.run(u0=P.u_exact(0.0), t0=0.0, Tend=Tend)
    return P, uend


@pytest.mark.firedrake
@pytest.mark.parametrize('name', ['heat', 'convdiff'])
def test_reference_values(name):
    """The same steps give the recorded values, to round-off."""
    import firedrake as fd
    import numpy as np

    P, uend = run(name, DT[name], 2 * DT[name], maxiter=3, restol=-1.0)
    values = fd.PointEvaluator(P.mesh, [list(x) for x in REFERENCE[name]]).evaluate(uend.functionspace)
    assert np.allclose(values, list(REFERENCE[name].values()), rtol=0, atol=1e-12), values


@pytest.mark.firedrake
def test_heat_accuracy():
    """The project's heat setup reaches its accuracy target, 1.54e-4."""
    P, uend = run('heat', 0.2, 1.0, maxiter=20, restol=1e-11)
    uex = P.u_exact(1.0)
    assert abs(uex - uend) / abs(uex) <= 2e-4


@pytest.mark.firedrake
def test_convdiff_exact_solution():
    """The rotating Gaussian solves the equation without forcing, as the class assumes: u_t + U.grad u = nu lap u."""
    import firedrake as fd
    from pySDC.helpers.ufl_forms import time_derivative
    from pySDC.projects.StroemungsRaum.problem_classes.ConvectionDiffusion_2D_Firedrake import ConvDiff2DFiredrake

    P = ConvDiff2DFiredrake(c_nvars=8)
    x, y = fd.SpatialCoordinate(P.mesh)
    u = P.u_exact_expr
    residual = time_derivative(u, P.t) + fd.dot(fd.as_vector([-4 * y, 4 * x]), fd.grad(u)) - P.nu * fd.div(fd.grad(u))
    for t in [0.0, 0.3]:
        P.t.assign(t)
        size = fd.assemble(residual**2 * fd.dx(degree=12)) ** 0.5
        assert size < 1e-10 * fd.assemble(fd.dot(fd.grad(u), fd.grad(u)) * fd.dx(degree=12)) ** 0.5, size
