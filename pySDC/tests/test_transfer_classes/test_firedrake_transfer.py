import pytest


@pytest.mark.firedrake
def test_Firedrake_transfer():
    import firedrake as fd
    from firedrake import interpolate
    from pySDC.implementations.problem_classes.HeatFiredrake import Heat1DForcedFiredrake
    from pySDC.implementations.transfer_classes.TransferFiredrakeMesh import MeshToMeshFiredrake
    import numpy as np

    # prepare fine and coarse problems
    resolutions = [2, 4]
    probs = [Heat1DForcedFiredrake(n=n) for n in resolutions]

    # prepare data that we can interpolate exactly in this discretization
    functions = [fd.Function(me.V).interpolate(me.x**2) for me in probs]
    data = [me.u_init for me in probs]
    [data[i].assign(functions[i]) for i in range(len(functions))]
    rhs = [me.f_init for me in probs]
    [rhs[i].impl.assign(functions[i]) for i in range(len(functions))]
    [rhs[i].expl.assign(functions[i]) for i in range(len(functions))]

    # setup transfer class
    transfer = MeshToMeshFiredrake(*probs[::-1], {})

    # test that restriction and interpolation were indeed exact on the mesh
    restriction = transfer.restrict(data[1])
    error = abs(restriction - data[0])
    assert np.isclose(error, 0), f'Got unexpectedly large {error=} during restriction!'

    interpolation = transfer.prolong(data[0])
    error = abs(interpolation - data[1])
    assert np.isclose(error, 0), f'Got unexpectedly large {error=} during interpolation!'

    # test that restriction and interpolation were indeed exact on the IMEX mesh
    restriction = transfer.restrict(rhs[1])
    error = max([abs(restriction.impl - rhs[0].impl), abs(restriction.expl - rhs[0].expl)])
    assert np.isclose(error, 0), f'Got unexpectedly large {error=} during restriction!'

    interpolation = transfer.prolong(rhs[0])
    error = max([abs(interpolation.impl - rhs[1].impl), abs(interpolation.expl - rhs[1].expl)])
    assert np.isclose(error, 0), f'Got unexpectedly large {error=} during interpolation!'


@pytest.mark.firedrake
def test_Firedrake_transfer_hierarchy():
    import firedrake as fd
    from firedrake import interpolate
    from pySDC.implementations.problem_classes.HeatFiredrake import Heat1DForcedFiredrake
    from pySDC.implementations.transfer_classes.TransferFiredrakeMesh import MeshToMeshFiredrakeHierarchy
    import numpy as np

    # prepare fine and coarse problems with the hierarchy
    _prob = Heat1DForcedFiredrake(n=2)
    hierarchy = fd.MeshHierarchy(_prob.mesh, 1)
    probs = [Heat1DForcedFiredrake(mesh=mesh) for mesh in hierarchy]

    # prepare data that we can interpolate exactly in this discretization
    functions = [fd.Function(me.V).interpolate(me.x**2) for me in probs]
    data = [me.u_init for me in probs]
    [data[i].assign(functions[i]) for i in range(len(functions))]
    rhs = [me.f_init for me in probs]
    [rhs[i].impl.assign(functions[i]) for i in range(len(functions))]
    [rhs[i].expl.assign(functions[i]) for i in range(len(functions))]

    # setup transfer class
    transfer = MeshToMeshFiredrakeHierarchy(*probs[::-1], {})

    # test that restriction and interpolation were indeed exact on the mesh
    restriction = transfer.restrict(data[1])
    error = abs(restriction - data[0])
    assert np.isclose(error, 0), f'Got unexpectedly large {error=} during restriction!'

    interpolation = transfer.prolong(data[0])
    error = abs(interpolation - data[1])
    assert np.isclose(error, 0), f'Got unexpectedly large {error=} during interpolation!'

    # test that restriction and interpolation were indeed exact on the IMEX mesh
    restriction = transfer.restrict(rhs[1])
    error = max([abs(restriction.impl - rhs[0].impl), abs(restriction.expl - rhs[0].expl)])
    assert np.isclose(error, 0), f'Got unexpectedly large {error=} during restriction!'

    interpolation = transfer.prolong(rhs[0])
    error = max([abs(interpolation.impl - rhs[1].impl), abs(interpolation.expl - rhs[1].expl)])
    assert np.isclose(error, 0), f'Got unexpectedly large {error=} during interpolation!'


@pytest.mark.firedrake
def test_Firedrake_transfer_hierarchy_mixed():
    """Mixed function spaces, as Gusto uses, are transferred exactly as well."""
    import firedrake as fd
    from pySDC.implementations.datatype_classes.firedrake_mesh import firedrake_mesh
    from pySDC.implementations.transfer_classes.TransferFiredrakeMesh import MeshToMeshFiredrakeHierarchy
    import numpy as np

    hierarchy = fd.MeshHierarchy(fd.UnitIntervalMesh(2), 1)
    spaces = [fd.FunctionSpace(mesh, 'CG', 2) * fd.FunctionSpace(mesh, 'CG', 1) for mesh in hierarchy]
    data = []
    for W in spaces:
        u = firedrake_mesh(W)
        (x,) = fd.SpatialCoordinate(W.mesh())
        u.functionspace.subfunctions[0].interpolate(x**2)
        u.functionspace.subfunctions[1].interpolate(1 - x)
        data.append(u)

    coarse, fine = firedrake_mesh(spaces[0]), firedrake_mesh(spaces[1])
    MeshToMeshFiredrakeHierarchy._restrict(u_fine=data[1], u_coarse=coarse)
    MeshToMeshFiredrakeHierarchy._prolong(u_coarse=data[0], u_fine=fine)
    for got, expect in [(coarse, data[0]), (fine, data[1])]:
        assert np.isclose(abs(got - expect), 0), 'Mixed function was not transferred exactly'

    # Gusto's spaces on the sphere: plain `firedrake.inject` gives NaN for BDM on an immersed manifold
    hierarchy = fd.MeshHierarchy(fd.IcosahedralSphereMesh(1.0, 1, degree=2), 1)
    spaces = []
    for mesh in hierarchy:
        mesh.init_cell_orientations(fd.SpatialCoordinate(mesh))
        spaces.append(fd.FunctionSpace(mesh, 'BDM', 2) * fd.FunctionSpace(mesh, 'DG', 1))
    fine = firedrake_mesh(spaces[1])
    x = fd.SpatialCoordinate(hierarchy[1])
    fine.functionspace.subfunctions[0].project(fd.as_vector([-x[1], x[0], 0]))
    coarse = firedrake_mesh(spaces[0])
    MeshToMeshFiredrakeHierarchy._restrict(u_fine=fine, u_coarse=coarse)
    velocity = coarse.functionspace.subfunctions[0]
    assert np.isfinite(velocity.dat.data_ro).all(), 'Injection of BDM on the sphere gave non-finite values'
    assert np.isclose(fd.norm(velocity), fd.norm(fine.functionspace.subfunctions[0]), rtol=0.1)


if __name__ == '__main__':
    test_Firedrake_transfer_hierarchy()
