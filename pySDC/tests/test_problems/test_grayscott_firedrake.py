"""
Tests for the Gray-Scott problem written as forms for Firedrake, a system with two unknowns on a mixed space.
"""

import pytest

SWEEPER_PARAMS = {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'LU'}
SOLVER_PARAMS = {'snes_rtol': 1e-13, 'snes_atol': 1e-14}

# u and v at x = 40, 45, 50 after two steps of dt = 1 with exactly four SDC iterations each (256 cells, CG4 with
# Firedrake's default nodes, Newton tolerances 1e-13), computed with `GrayScott1DFiredrake` itself (Firedrake 2026.10),
# as a check against unintended changes
REFERENCE = {
    40.0: (0.9899250073282054, 0.0014233336355883224),
    45.0: (0.842731657845967, 0.06899513142051854),
    50.0: (0.5834120589275994, 0.2804413258641244),
}


def run(problem_params, maxiter, restol, nsteps, levels=1, **description):
    """Run (ML)SDC on Gray-Scott and return the end point and the iterations per step."""
    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.problem_classes.GrayScott_1D_Firedrake import GrayScott1DFiredrake
    from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass

    description = {
        'problem_class': GrayScott1DFiredrake,
        'problem_params': problem_params,
        'sweeper_class': generic_implicit_mass,
        'sweeper_params': SWEEPER_PARAMS,
        'level_params': {'dt': 1.0, 'restol': restol},
        'step_params': {'maxiter': maxiter},
        **description,
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    P = controller.MS[0].levels[0].prob
    uend, stats = controller.run(u0=P.u_exact(0.0), t0=0.0, Tend=float(nsteps))
    return uend, np.array([me[1] for me in get_sorted(stats, type='niter')])


@pytest.mark.firedrake
def test_reference_values():
    """The recorded values, to round-off."""
    import firedrake as fd
    import numpy as np

    params = {'n': 256, 'order': 4, 'solver_parameters': SOLVER_PARAMS}
    uend, _ = run(params, maxiter=4, restol=-1.0, nsteps=2)

    points = [[x] for x in REFERENCE]
    u, v = uend.functionspace.subfunctions
    evaluate = fd.PointEvaluator(u.function_space().mesh(), points).evaluate
    expect = np.array(list(REFERENCE.values()))
    assert np.allclose(np.stack([evaluate(u), evaluate(v)], axis=1), expect, rtol=0, atol=1e-11)


@pytest.mark.firedrake
def test_multilevel_on_mixed_space():
    """
    MLSDC with two and three levels on a mesh hierarchy converges to the SDC solution in fewer iterations, so the
    mixed space goes through the dual data, the transfers and the solves unchanged.
    """
    import firedrake as fd
    import numpy as np
    from pySDC.implementations.transfer_classes.BaseTransfer_mass import base_transfer_mass
    from pySDC.implementations.transfer_classes.TransferFiredrakeMesh import MeshToMeshFiredrakeHierarchy

    meshes = list(fd.MeshHierarchy(fd.IntervalMesh(32, 100), 2))[::-1]
    transfer = {'base_transfer_class': base_transfer_mass, 'space_transfer_class': MeshToMeshFiredrakeHierarchy}

    def go(levels):
        params = {'mesh': meshes[:levels], 'solver_parameters': SOLVER_PARAMS}
        return run(params, maxiter=30, restol=1e-8, nsteps=2, **transfer)

    u_sdc, iterations_sdc = go(1)
    for levels in [2, 3]:
        u, iterations = go(levels)
        assert iterations.mean() < iterations_sdc.mean(), (levels, iterations, iterations_sdc)
        difference = max(np.abs(a - b).max() for a, b in zip(u.dat.data_ro, u_sdc.dat.data_ro, strict=True))
        assert difference < 1e-7, (levels, difference)


@pytest.mark.firedrake
def test_initial_value_only():
    """Gray-Scott has no exact solution, only an initial value."""
    from pySDC.implementations.problem_classes.GrayScott_1D_Firedrake import GrayScott1DFiredrake

    P = GrayScott1DFiredrake(n=8, order=2)
    assert P.u_exact(0).is_dual is False
    with pytest.raises(AssertionError):
        P.u_exact(1.0)
