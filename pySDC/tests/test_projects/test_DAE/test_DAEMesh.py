import pytest


@pytest.mark.base
@pytest.mark.parametrize('shape', [(6,), (4, 6)])
def testInitialization(shape):
    """
    Tests for a random init if initialization results in desired shape of mesh.
    """

    import numpy as np
    from pySDC.projects.DAE.misc.DAEMesh import DAEMesh

    init = (shape, None, np.dtype('float64'))
    mesh = DAEMesh(init)

    for comp in mesh.components:
        assert comp in dir(mesh), f'ERROR: DAEMesh does not have a {comp} attribute!'
        print(mesh.__dict__[comp].size)
        assert np.shape(mesh.__dict__[comp]) == shape, f'ERROR: Component {comp} does not have the desired length!'

    assert len(mesh.components) == len(mesh), 'ERROR: Mesh does not contain two component arrays!'


@pytest.mark.base
def testInitializationGivenMesh():
    """
    Tests if for a given mesh the initialization results in the same mesh.
    """

    import numpy as np
    from pySDC.projects.DAE.misc.DAEMesh import DAEMesh

    nvars_1d = 6
    init = (nvars_1d, None, np.dtype('float64'))
    mesh1 = DAEMesh(init)
    mesh1.diff[:] = np.arange(6)
    mesh1.alg[:] = np.arange(6, 12)

    mesh2 = DAEMesh(mesh1)

    assert np.allclose(mesh1.diff, mesh2.diff) and np.allclose(
        mesh1.alg, mesh2.alg
    ), 'ERROR: Components in initialized meshes do not match!'


@pytest.mark.base
def testArrayUFuncOperator():
    """
    Test if overloaded __array_ufunc__ operator of datatype does what it is supposed to do.
    """

    import numpy as np
    from pySDC.projects.DAE.misc.DAEMesh import DAEMesh

    init = ((4, 6), None, np.dtype('float64'))
    mesh = DAEMesh(init)

    # addition with numpy array
    mesh += np.arange(6)

    for comp in mesh.components:
        assert comp in dir(mesh), f'ERROR: After addition DAEMesh has lost {comp} attribute!'

        assert np.allclose(mesh.__dict__[comp], np.arange(6)), 'ERROR: Addition did not provide desired result!'

    # addition with mesh type
    mesh2 = DAEMesh(mesh)
    mesh += mesh2

    # for comp in mesh.components:
        # assert comp in dir(mesh), f'ERROR: After addition DAEMesh has lost {comp} attribute!'

        # assert np.allclose(mesh.__dict__[comp], 2 * np.arange(6)), 'ERROR: Addition did not provide desired result!'
