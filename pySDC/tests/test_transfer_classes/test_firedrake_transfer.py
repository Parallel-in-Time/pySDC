import pytest


@pytest.mark.firedrake
def test_Firedrake_transfer():
    import firedrake as fd
    from firedrake.__future__ import interpolate
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
    from firedrake.__future__ import interpolate
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


if __name__ == '__main__':
    test_Firedrake_transfer_hierarchy()
