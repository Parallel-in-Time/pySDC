import pytest


@pytest.mark.base
@pytest.mark.parametrize('shape', [(6,), (4, 6), (4, 6, 8)])
def testInitialization(shape):
    """
    Tests for a random init if initialization results in desired shape of mesh.
    """

    import numpy as np
    from pySDC.projects.DAE.misc.DAEMesh import DAEMesh

    init = (shape, None, np.dtype('float64'))
    mesh = DAEMesh(init)

    assert np.shape(mesh.diff) == shape, f'ERROR: Component diff does not have the desired length!'
    assert np.shape(mesh.alg) == shape, f'ERROR: Component alg does not have the desired length!'

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
@pytest.mark.parametrize('shape', [(6,), (4, 6), (4, 6, 8)])
def testArrayUFuncOperator(shape):
    """
    Test if overloaded __array_ufunc__ operator of datatype does what it is supposed to do.
    """

    import numpy as np
    from pySDC.projects.DAE.misc.DAEMesh import DAEMesh

    init = (shape, None, np.dtype('float64'))
    mesh = DAEMesh(init)
    mesh2 = DAEMesh(mesh)

    randomArr = np.random.random(shape)
    mesh.diff[:] = randomArr
    mesh2.diff[:] = 2 * randomArr

    subMesh = mesh - mesh2
    assert type(subMesh) == DAEMesh
    assert np.allclose(subMesh.diff, randomArr - 2 * randomArr)
    assert np.allclose(subMesh.alg, 0)

    addMesh = mesh + mesh2
    assert type(addMesh) == DAEMesh
    assert np.allclose(addMesh.diff, randomArr + 2 * randomArr)
    assert np.allclose(addMesh.alg, 0)

    sinMesh = np.sin(mesh)
    assert type(sinMesh) == DAEMesh
    assert np.allclose(sinMesh.diff, np.sin(randomArr))
    assert np.allclose(sinMesh.alg, 0)
