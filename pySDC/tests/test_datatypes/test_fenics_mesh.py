import pytest


def meshes():
    import dolfin as df
    import numpy as np
    from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh

    V = df.FunctionSpace(df.UnitIntervalMesh(16), 'CG', 1)
    rng = np.random.default_rng(0)
    a, b = fenics_mesh(V), fenics_mesh(V)
    a.values.vector()[:] = rng.random(17)
    b.values.vector()[:] = rng.random(17)
    return a, b


def values(x):
    return x.values.vector().get_local()


@pytest.mark.fenics
def test_operators_make_new_meshes():
    """`+`, `-` and scaling leave their operands alone and give what the arithmetic on the values gives."""
    import numpy as np

    a, b = meshes()
    va, vb = values(a), values(b)

    for result, expected in [(a + b, va + vb), (a - b, va - vb), (0.3 * a, 0.3 * va)]:
        assert result is not a and result is not b
        assert np.array_equal(values(result), expected)
    assert np.array_equal(values(a), va) and np.array_equal(values(b), vb), 'an operand was changed'


@pytest.mark.fenics
def test_in_place_operators():
    """`+=` and `-=` change the left operand itself, and only that."""
    import numpy as np

    a, b = meshes()
    va, vb = values(a), values(b)
    alias = a

    a += b
    assert a is alias and np.array_equal(values(a), va + vb)
    a -= 2.0 * b
    assert a is alias and np.array_equal(values(a), va + vb - 2.0 * vb)
    assert np.array_equal(values(b), vb), 'the right operand was changed'
