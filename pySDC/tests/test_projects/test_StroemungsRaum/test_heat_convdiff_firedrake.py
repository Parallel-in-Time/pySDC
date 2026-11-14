"""
Tests for the Firedrake ports of StroemungsRaum's 2D heat and convection-diffusion problems. They live here rather
than in the project, because the project's own tests run in its FEniCS environment, and these need Firedrake.
"""

import pytest

# Two steps from t = 0 with two RADAU-RIGHT nodes, the default preconditioners and exactly three iterations, computed
# with `fenics_heat2D_mass` (dt = 0.2) and `fenics_ConvDiff2D_mass` (dt = 0.005) with their default parameters, as
# u at the points. Kept here so the comparison survives FEniCS.
FENICS = {
    'heat': {
        (0.1, 0.2): 0.1673141704633517,
        (0.5, 0.5): 0.9211559269793309,
        (0.73, 0.31): 0.5714878163684806,
        (0.9, 0.85): 0.12922964366379383,
    },
    'convdiff': {
        (-0.25, 0.0): 0.8330362833718628,
        (-0.2, 0.05): 0.1063283232246961,
        (-0.3, -0.03): 0.3150474007899712,
        (-0.27, 0.02): 0.5482890607334171,
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
def test_reproduces_FEniCS(name):
    """The same steps as the FEniCS version, on the same mesh, give the same solution to round-off."""
    import firedrake as fd
    import numpy as np

    P, uend = run(name, DT[name], 2 * DT[name], maxiter=3, restol=-1.0)
    values = fd.PointEvaluator(P.mesh, [list(x) for x in FENICS[name]]).evaluate(uend.functionspace)
    assert np.allclose(values, list(FENICS[name].values()), rtol=0, atol=1e-12), values


@pytest.mark.firedrake
def test_heat_accuracy():
    """The project's heat setup reaches the accuracy the FEniCS test asks for (1.54e-4 there, as here)."""
    P, uend = run('heat', 0.2, 1.0, maxiter=20, restol=1e-11)
    uex = P.u_exact(1.0)
    assert abs(uex - uend) / abs(uex) <= 2e-4


@pytest.mark.firedrake
def test_convdiff_exact_solution():
    """The rotating Gaussian solves the equation without forcing, as the FEniCS version assumes: u_t + U.grad u = nu lap u."""
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
