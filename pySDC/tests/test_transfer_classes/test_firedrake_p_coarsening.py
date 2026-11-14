"""
p-coarsening with Firedrake: lower-degree Lagrange spaces on the same mesh as coarse levels, transferred by
`MeshToMeshFiredrake` (interpolation, its adjoint for load vectors, and L2 projection for solutions).
"""

import pytest


@pytest.mark.firedrake
@pytest.mark.parametrize('family', ['CG', 'DG'])
def test_transfers_are_exact(family):
    """P^T gives the coarse load vector of the same functional, and projection returns a prolonged coarse function."""
    from types import SimpleNamespace

    import firedrake as fd
    import numpy as np
    from pySDC.core.errors import TransferError
    from pySDC.implementations.datatype_classes.firedrake_mesh import firedrake_mesh
    from pySDC.implementations.transfer_classes.TransferFiredrakeMesh import MeshToMeshFiredrake

    mesh = fd.UnitIntervalMesh(16)
    x = fd.SpatialCoordinate(mesh)[0]
    fine, coarse = [
        SimpleNamespace(init=fd.FunctionSpace(mesh, family, p, variant='equispaced'), dtype_u=firedrake_mesh)
        for p in (4, 2)
    ]
    transfer = MeshToMeshFiredrake(fine, coarse, {})

    def load_vector(V):
        return firedrake_mesh(fd.assemble(fd.exp(x) * fd.cos(5 * x) * fd.TestFunction(V) * fd.dx(degree=20)))

    restricted = transfer.restrict_dual(load_vector(fine.init))
    assert restricted.is_dual
    assert np.allclose(restricted.dat.data_ro, load_vector(coarse.init).dat.data_ro, rtol=0, atol=1e-15)

    u_coarse = firedrake_mesh(fd.Function(coarse.init).interpolate(fd.sin(3 * x)))
    projected = transfer.project(transfer.prolong(u_coarse))
    assert not projected.is_dual
    assert abs(projected - u_coarse) < 1e-13 * abs(u_coarse)

    with pytest.raises(TransferError):
        transfer.restrict_dual(u_coarse)
    with pytest.raises(TransferError):
        transfer.project(restricted)


def run(problem_class, problem_params, sweeper_class, orders, dt, nsteps, restol):
    """Run (ML)SDC with the given degrees on one mesh, coarsest last, and return the end point and the iterations."""
    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.transfer_classes.BaseTransfer_mass import base_transfer_mass
    from pySDC.implementations.transfer_classes.TransferFiredrakeMesh import MeshToMeshFiredrake

    description = {
        'problem_class': problem_class,
        # the mesh is shared by all levels, the degree is one per level
        'problem_params': {**problem_params, 'order': orders},
        'sweeper_class': sweeper_class,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'LU'},
        'level_params': {'dt': dt, 'restol': restol},
        'step_params': {'maxiter': 50},
        'base_transfer_class': base_transfer_mass,
        'space_transfer_class': MeshToMeshFiredrake,
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    P = controller.MS[0].levels[0].prob
    uend, stats = controller.run(u0=P.u_exact(0.0), t0=0.0, Tend=nsteps * dt)
    return uend, np.array([me[1] for me in get_sorted(stats, type='niter')])


@pytest.mark.firedrake
@pytest.mark.parametrize('example', ['heat', 'grayscott'])
def test_mlsdc_in_degree_matches_sdc(example):
    """
    MLSDC coarsening in the degree on one mesh converges to the SDC solution: the heat equation with time-dependent
    boundary values (two and three levels), and Gray-Scott on a mixed space.
    """
    import firedrake as fd
    import numpy as np

    if example == 'heat':
        from pySDC.implementations.problem_classes.HeatEquation_1D_FormFiredrake import Heat1DMassFiredrake as cls
        from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass as sweeper

        params = {'mesh': fd.UnitIntervalMesh(16), 'nu': 0.1, 'c': 1.0, 'time_dependent_bc': True}
        ladders, dt, restol = [[4, 2], [4, 2, 1]], 0.1, 1e-11
    else:
        from pySDC.implementations.problem_classes.GrayScott_1D_Firedrake import GrayScott1DFiredrake as cls
        from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass as sweeper

        params = {'mesh': fd.IntervalMesh(64, 100), 'solver_parameters': {'snes_rtol': 1e-13, 'snes_atol': 1e-14}}
        ladders, dt, restol = [[4, 2]], 1.0, 1e-8

    u_sdc, iterations_sdc = run(cls, params, sweeper, [4], dt, 3, restol)
    for orders in ladders:
        u, iterations = run(cls, params, sweeper, orders, dt, 3, restol)
        difference = max(np.abs(a - b).max() for a, b in zip(u.dat.data_ro, u_sdc.dat.data_ro, strict=True))
        assert difference < 100 * restol, (orders, difference)
        assert iterations.max() < 50, (orders, iterations)
