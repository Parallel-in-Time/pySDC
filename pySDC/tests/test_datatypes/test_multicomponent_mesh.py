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


@pytest.mark.base
def test_component_properties():
    from pySDC.implementations.datatype_classes.mesh import MultiComponentMesh, mesh
    import numpy as xp

    single_property_test(xp, MultiComponentMesh, mesh)


@pytest.mark.cupy
def test_CuPy_component_properties():
    from pySDC.implementations.datatype_classes.cupy_mesh import CuPyMultiComponentMesh, cupy_mesh
    import cupy as xp

    single_property_test(xp, CuPyMultiComponentMesh, cupy_mesh)


def single_property_test(xp, MultiComponentMeshClass, base):
    import copy

    class TestMesh(MultiComponentMeshClass):
        components = ['a', 'b']

    A = TestMesh(((4,), None, xp.dtype('d')))

    # assigning a component without `[:]` has to write into the mesh rather than shadow it with a new attribute
    A.a = 1.0
    A.b[:] = 2.0
    assert xp.allclose(A[0], 1.0), 'Assignment without `[:]` did not reach the mesh!'
    assert xp.shares_memory(A, A.a)

    # ... and therefore has to survive every operation that drops instance attributes
    for B in [copy.deepcopy(A), A.copy(), 1.0 * A, TestMesh(A)]:
        assert type(B) is TestMesh, f'Lost the type in {type(B)}!'
        assert xp.allclose(B.a, 1.0) and xp.allclose(B.b, 2.0), f'Lost the components in {type(B)}!'

    A.a -= 1.0
    assert xp.allclose(A[0], 0.0), 'In-place operation on a component did not reach the mesh!'

    # a component is a view of the single-component datatype, not of the multi-component one
    assert type(A.a) is base, f'Expected a component of type {base}, got {type(A.a)}!'

    # components may only be accessed while the leading axis still counts them
    with pytest.raises(AttributeError):
        _ = A[:1].a

    with pytest.raises(AttributeError):
        _ = A.not_a_component

    # a component that shadows an attribute of the base class has to be refused when the class is made
    with pytest.raises(AttributeError):

        class ClashingMesh(MultiComponentMeshClass):
            components = ['xp', 'u']


if __name__ == '__main__':
    test_MultiComponentMesh(1)
