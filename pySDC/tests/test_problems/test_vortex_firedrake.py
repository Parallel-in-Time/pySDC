"""
Tests for the periodic vorticity problem with Firedrake (`Vortex2DFiredrake`), with the stream function from a Poisson
solve in every right-hand side evaluation.
"""

import pytest

POINTS = [[0.1, 0.25], [0.3, 0.7], [0.55, 0.26], [0.8, 0.5]]
# the vorticity at POINTS after two steps of dt = 0.001 (32 x 32 squares with Firedrake's default diagonal, CG2, nu = 0.01, rho = 50, delta = 0.05, three
# RADAU-RIGHT nodes, LU, exactly three iterations), computed with `Vortex2DFiredrake` itself (Firedrake 2026.10), as a
# check against unintended changes. It no longer matches FEniCS, whose advection has the opposite sign.
REFERENCE = [44.811970527860026, -1.1220729397285363, 35.09452421651993, -0.29732767768767804]


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
def test_reference_values():
    """The Poisson-solve variant still gives the recorded values, to round-off."""
    import firedrake as fd
    import numpy as np
    from pySDC.implementations.problem_classes.VorticityVelocity_2D_Firedrake_periodic import Vortex2DFiredrake

    mesh = fd.PeriodicUnitSquareMesh(32, 32)
    w, _ = run(Vortex2DFiredrake, [mesh], maxiter=3, restol=-1.0, nsteps=2)
    values = fd.PointEvaluator(mesh, POINTS).evaluate(w)
    assert np.allclose(values, REFERENCE, rtol=0, atol=1e-11), values


@pytest.mark.firedrake
def test_mlsdc_matches_sdc():
    """
    MLSDC on two meshes of a hierarchy converges to the SDC solution, in no more iterations. It saves none here: the
    coarse level does not pay for itself on the vortex, see the FEM study.
    """
    import firedrake as fd
    import numpy as np
    from pySDC.implementations.problem_classes.VorticityVelocity_2D_Firedrake_periodic import Vortex2DFiredrake

    meshes = list(fd.MeshHierarchy(fd.PeriodicUnitSquareMesh(8, 8), 1))[::-1]
    w_sdc, iterations_sdc = run(Vortex2DFiredrake, meshes[:1], maxiter=30, restol=1e-9, nsteps=3)
    w_mlsdc, iterations_mlsdc = run(Vortex2DFiredrake, meshes, maxiter=30, restol=1e-9, nsteps=3)

    assert np.mean(iterations_mlsdc) <= np.mean(iterations_sdc), (iterations_mlsdc, iterations_sdc)
    difference = np.abs(w_mlsdc.dat.data_ro - w_sdc.dat.data_ro).max() / np.abs(w_sdc.dat.data_ro).max()
    assert difference < 1e-8, difference


@pytest.mark.firedrake
def test_initial_value_only():
    """The vortex has no exact solution, only an initial value."""
    from pySDC.implementations.problem_classes.VorticityVelocity_2D_Firedrake_periodic import Vortex2DFiredrake

    with pytest.raises(AssertionError):
        Vortex2DFiredrake(n=8).u_exact(0.1)
