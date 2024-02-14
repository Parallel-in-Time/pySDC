import pytest


@pytest.mark.base
def testInitialization():
    """
    Tests for a random init if initialization results in desired shape of mesh.
    """

    import numpy as np
    from pySDC.projects.DAE.misc.DAEMesh import DAEMesh

    init = (6, None, np.dtype('float64'))
    mesh1 = DAEMesh(init)
    mesh1.diff[:] = np.arange(6)
    mesh1.alg[:] = np.arange(6, 12)
    mesh2 = DAEMesh(mesh1)

    for mesh in [mesh1, mesh2]:
        assert 'diff' in dir(mesh), 'ERROR: DAEMesh does not have a diff attribute!'
        assert 'alg' in dir(mesh), 'ERROR: DAEMesh does not have a diff attribute!'

        assert len(mesh.diff) == 6 and len(mesh.alg) == 6, 'ERROR: Components does not have the desired length!'

        assert len(mesh.components) == len(mesh), 'ERROR: Mesh does not contain two component arrays!'

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

    init = (6, None, np.dtype('float64'))
    mesh = DAEMesh(init)

    mesh += np.arange(6)
    for m in range(6):
        assert mesh.diff[m] == m and mesh.alg[m] == m, 'ERROR: Addition did not provide desired result!'

    assert 'diff' in dir(mesh), 'ERROR: After addition DAEMesh has lost diff attribute!'
    assert 'alg' in dir(mesh), 'ERROR: After addition DAEMesh has lost alg attribute!'
