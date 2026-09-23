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
def test_component_assignment_writes_into_the_mesh():
    """Assigning a component without ``[:]`` must write into the mesh rather than shadow it with a new attribute."""
    from pySDC.implementations.datatype_classes.mesh import MultiComponentMesh
    import numpy as np
    import copy

    class TestMesh(MultiComponentMesh):
        components = ['a', 'b']

    A = TestMesh(((4,), None, np.dtype('d')))
    A.a = 1.0
    A.b[:] = 2.0

    assert np.allclose(np.asarray(A)[0], 1.0), 'Assignment without `[:]` did not reach the mesh!'
    assert np.shares_memory(A, A.a)

    # the value has to survive anything that drops instance attributes
    for B in [copy.deepcopy(A), 1.0 * A, TestMesh(A)]:
        assert np.allclose(B.a, 1.0) and np.allclose(B.b, 2.0), f'Lost the components in {type(B)}!'

    A.a -= 1.0
    assert np.allclose(np.asarray(A)[0], 0.0), 'In-place operation on a component did not reach the mesh!'


@pytest.mark.base
def test_component_name_clash_is_caught():
    """A component that shadows an attribute of the base class has to be refused when the class is made."""
    from pySDC.implementations.datatype_classes.mesh import MultiComponentMesh

    with pytest.raises(AttributeError):

        class ClashingMesh(MultiComponentMesh):
            components = ['T', 'u']


@pytest.mark.base
def test_component_access_with_unexpected_shape():
    """Components may only be accessed if the leading axis still counts the components."""
    from pySDC.implementations.datatype_classes.mesh import MultiComponentMesh
    import numpy as np

    class TestMesh(MultiComponentMesh):
        components = ['a', 'b']

    A = TestMesh(((4,), None, np.dtype('d')))

    with pytest.raises(AttributeError):
        A[:1].a

    with pytest.raises(AttributeError):
        A.not_a_component


if __name__ == '__main__':
    test_MultiComponentMesh(1)
