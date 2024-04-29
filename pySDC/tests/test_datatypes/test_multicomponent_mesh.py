import pytest


@pytest.mark.base
@pytest.mark.parametrize('shape', [1, (3,), (2, 4)])
def test_MultiComponentMesh(shape):
    from pySDC.implementations.datatype_classes.mesh import MultiComponentMesh as MultiComponentMeshClass
    import numpy as xp

    single_test(shape, xp, MultiComponentMeshClass)


@pytest.mark.cupy
@pytest.mark.parametrize('shape', [1, (3,), (2, 4)])
def test_CuPyMultiComponentMesh(shape):
    from pySDC.implementations.datatype_classes.cupy_mesh import CuPyMultiComponentMesh as MultiComponentMeshClass
    import cupy as xp

    single_test(shape, xp, MultiComponentMeshClass)


def single_test(shape, xp, MultiComponentMeshClass):
    class TestMesh(MultiComponentMeshClass):
        components = ['a', 'b']

    # instantiate meshes
    init = (shape, None, xp.dtype('D'))
    A = TestMesh(init)
    B = TestMesh(A)

    # fill part of the meshes with values
    a = xp.random.random(shape)
    b = xp.random.random(shape)
    zero = xp.zeros_like(a)
    A.a[:] = a
    B.a[:] = b

    # check that the meshes have been prepared appropriately
    for M, m in zip([A, B], [a, b]):
        assert M.shape == (len(TestMesh.components),) + ((shape,) if type(shape) is int else shape)
        assert xp.allclose(M.a, m)
        assert xp.allclose(M.b, zero)
        assert xp.shares_memory(M, M.a)
        assert xp.shares_memory(M, M.b)
        assert not xp.shares_memory(M.a, m)

    # check that various computations give the desired results
    assert xp.allclose(A.a + B.a, a + b)
    assert xp.allclose((A + B).a, a + b)
    assert xp.allclose((A + B).b, zero)

    C = A - B
    assert xp.allclose(C.a, a - b)
    assert xp.allclose(C.b, zero)
    assert not xp.shares_memory(A, C)
    assert not xp.shares_memory(B, C)

    D = xp.exp(A)
    assert type(D) == TestMesh
    assert xp.allclose(D.a, xp.exp(a))
    assert xp.allclose(D.b, zero + 1)
    assert not xp.shares_memory(A, D)

    B *= A
    assert xp.allclose(B.a, a * b)
    assert xp.allclose(A.a, a)
    assert xp.allclose(B.b, zero)
    assert xp.allclose(A.b, zero)
    assert not xp.shares_memory(A, B)

    A /= 10.0
    assert xp.allclose(A.a, a / 10)
    assert xp.allclose(A.b, zero)


if __name__ == '__main__':
    test_MultiComponentMesh(1)
