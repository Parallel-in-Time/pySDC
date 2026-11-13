"""
Tests for the periodic vorticity problem with Firedrake, with the stream function from a Poisson solve in the
right-hand side (`Vortex2DFiredrake`) and as an unknown on a mixed space (`Vortex2DMixedFiredrake`).
"""

import pytest

POINTS = [[0.1, 0.25], [0.3, 0.7], [0.55, 0.26], [0.8, 0.5]]
# the vorticity at POINTS after two steps of dt = 0.001 (32 x 32 squares, CG2, nu = 0.01, rho = 50, delta = 0.05, three
# RADAU-RIGHT nodes, LU, exactly three iterations), computed with `fenics_vortex_2d_mass` from FEniCS 2019.1 on the
# unrefined 32 x 32 mesh. Kept here so the comparison survives FEniCS.
FENICS_REFERENCE = [44.8119705278601, -1.0941718994856566, 34.72165742787179, -0.29972712561954923]


def run(problem_class, meshes, maxiter, restol, nsteps):
    """Run (ML)SDC from the double shear layer and return the end vorticity and the iterations per step."""
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
    from pySDC.implementations.transfer_classes.BaseTransfer_mass import base_transfer_mass
    from pySDC.implementations.transfer_classes.TransferFiredrakeMesh import MeshToMeshFiredrakeHierarchy

    description = {
        'problem_class': problem_class,
        'problem_params': {'mesh': meshes},
        'sweeper_class': imex_1st_order_mass,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'LU'},
        'level_params': {'dt': 0.001, 'restol': restol},
        'step_params': {'maxiter': maxiter},
        'base_transfer_class': base_transfer_mass,
        'space_transfer_class': MeshToMeshFiredrakeHierarchy,
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    P = controller.MS[0].levels[0].prob
    uend, stats = controller.run(u0=P.u_exact(0.0), t0=0.0, Tend=nsteps * 0.001)
    return uend.functionspace.subfunctions[0], [me[1] for me in get_sorted(stats, type='niter')]


@pytest.mark.firedrake
def test_reproduces_FEniCS():
    """The Poisson-solve variant reproduces the FEniCS version to round-off on the same mesh."""
    import firedrake as fd
    import numpy as np
    from pySDC.implementations.problem_classes.VorticityVelocity_2D_Firedrake_periodic import Vortex2DFiredrake

    mesh = fd.PeriodicUnitSquareMesh(32, 32, diagonal='right')
    w, _ = run(Vortex2DFiredrake, [mesh], maxiter=3, restol=-1.0, nsteps=2)
    values = fd.PointEvaluator(mesh, POINTS).evaluate(w)
    assert np.allclose(values, FENICS_REFERENCE, rtol=0, atol=1e-11), values


@pytest.mark.firedrake
@pytest.mark.parametrize('levels', [1, 2])
def test_mixed_matches_poisson_solve(levels):
    """
    With the stream function as an unknown, SDC and MLSDC take the same iterations and reach the same vorticity as
    with the stream function from a Poisson solve: both converge to the same collocation solution.
    """
    import firedrake as fd
    import numpy as np
    from pySDC.implementations.problem_classes.VorticityVelocity_2D_Firedrake_periodic import (
        Vortex2DFiredrake,
        Vortex2DMixedFiredrake,
    )

    meshes = list(fd.MeshHierarchy(fd.PeriodicUnitSquareMesh(8, 8, diagonal='right'), 1))[::-1][:levels]
    w_poisson, iterations_poisson = run(Vortex2DFiredrake, meshes, maxiter=30, restol=1e-9, nsteps=3)
    w_mixed, iterations_mixed = run(Vortex2DMixedFiredrake, meshes, maxiter=30, restol=1e-9, nsteps=3)

    assert iterations_mixed == iterations_poisson, (iterations_mixed, iterations_poisson)
    difference = np.abs(w_mixed.dat.data_ro - w_poisson.dat.data_ro).max() / np.abs(w_poisson.dat.data_ro).max()
    assert difference < 1e-9, difference


@pytest.mark.firedrake
def test_initial_value():
    """The mixed variant starts from the stream function of the initial vorticity; neither has later exact values."""
    import firedrake as fd
    import numpy as np
    from pySDC.implementations.problem_classes.VorticityVelocity_2D_Firedrake_periodic import (
        Vortex2DFiredrake,
        Vortex2DMixedFiredrake,
    )

    mixed = Vortex2DMixedFiredrake(n=8)
    w, psi = mixed.u_exact(0.0).functionspace.subfunctions
    q = fd.TestFunction(psi.function_space())
    residual = fd.assemble((fd.inner(fd.grad(psi), fd.grad(q)) - w * q) * fd.dx)
    assert np.abs(residual.dat.data_ro).max() < 1e-12

    for P in [mixed, Vortex2DFiredrake(n=8)]:
        with pytest.raises(AssertionError):
            P.u_exact(0.1)
