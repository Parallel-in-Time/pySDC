import pytest


@pytest.mark.base
@pytest.mark.parametrize('shape', [1, (3,), (2, 4)])
def test_MultiComponentMesh(shape):
    from pySDC.implementations.datatype_classes.mesh import MultiComponentMesh
    import numpy as np

    class TestMesh(MultiComponentMesh):
        components = ['a', 'b']

    # instantiate meshes
    init = (shape, None, np.dtype('D'))
    A = TestMesh(init)
    B = TestMesh(A)

    # fill part of the meshes with values
    a = np.random.random(shape)
    b = np.random.random(shape)
    zero = np.zeros_like(a)
    A.a[:] = a
    B.a[:] = b

    # check that the meshes have been prepared appropriately
    for M, m in zip([A, B], [a, b]):
        assert M.shape == (len(TestMesh.components),) + ((shape,) if type(shape) is int else shape)
        assert np.allclose(M.a, m)
        assert np.allclose(M.b, zero)
        assert np.shares_memory(M, M.a)
        assert np.shares_memory(M, M.b)
        assert not np.shares_memory(M.a, m)

    # check that various computations give the desired results
    assert np.allclose(A.a + B.a, a + b)
    assert np.allclose((A + B).a, a + b)
    assert np.allclose((A + B).b, zero)

    B *= A
    assert np.allclose(B.a, a * b)
    assert np.allclose(A.a, a)
    assert np.allclose(B.b, zero)
    assert np.allclose(A.b, zero)
    assert not np.shares_memory(A, B)

    A /= 10.0
    assert np.allclose(A.a, a / 10)
    assert np.allclose(A.b, zero)
