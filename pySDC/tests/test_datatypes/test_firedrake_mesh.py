import pytest


@pytest.mark.firedrake
def test_addition(n=3, v1=1, v2=2):
    from pySDC.implementations.datatype_classes.firedrake_mesh import firedrake_mesh
    import numpy as np
    import firedrake as fd

    mesh = fd.UnitSquareMesh(n, n)
    V = fd.VectorFunctionSpace(mesh, "CG", 2)

    a = firedrake_mesh(V)
    b = firedrake_mesh(a)

    a.assign(v1)
    b.assign(v2)

    c = a + b

    assert np.allclose(c.dat._numpy_data, v1 + v2)
    assert np.allclose(a.dat._numpy_data, v1)
    assert np.allclose(b.dat._numpy_data, v2)


@pytest.mark.firedrake
def test_subtraction(n=3, v1=1, v2=2):
    from pySDC.implementations.datatype_classes.firedrake_mesh import firedrake_mesh
    import numpy as np
    import firedrake as fd

    mesh = fd.UnitSquareMesh(n, n)
    V = fd.VectorFunctionSpace(mesh, "CG", 2)

    a = firedrake_mesh(V, val=v1)
    _b = fd.Function(V)
    _b.assign(v2)
    b = firedrake_mesh(_b)

    c = a - b

    assert np.allclose(c.dat._numpy_data, v1 - v2)
    assert np.allclose(a.dat._numpy_data, v1)
    assert np.allclose(b.dat._numpy_data, v2)


@pytest.mark.firedrake
def test_right_multiplication(n=3, v1=1, v2=2):
    from pySDC.implementations.datatype_classes.firedrake_mesh import firedrake_mesh
    from pySDC.core.errors import DataError
    import numpy as np
    import firedrake as fd

    mesh = fd.UnitSquareMesh(n, n)
    V = fd.VectorFunctionSpace(mesh, "CG", 2)

    a = firedrake_mesh(V)
    b = firedrake_mesh(a)

    a.assign(v1)

    b = v2 * a

    assert np.allclose(b.dat._numpy_data, v1 * v2)
    assert np.allclose(a.dat._numpy_data, v1)

    try:
        'Dat kölsche Dom' * b
    except DataError:
        pass


@pytest.mark.firedrake
def test_norm(n=3, v1=-1):
    from pySDC.implementations.datatype_classes.firedrake_mesh import firedrake_mesh
    import numpy as np
    import firedrake as fd

    mesh = fd.UnitSquareMesh(n, n)
    V = fd.VectorFunctionSpace(mesh, "CG", 1)

    a = firedrake_mesh(V, val=v1)
    b = firedrake_mesh(a)

    b = abs(a)

    assert np.isclose(b, np.sqrt(2) * abs(v1)), f'{b=}, {v1=}'
    assert np.allclose(a.dat._numpy_data, v1)


@pytest.mark.firedrake
def test_addition_rhs(n=3, v1=1, v2=2):
    from pySDC.implementations.datatype_classes.firedrake_mesh import IMEX_firedrake_mesh
    import numpy as np
    import firedrake as fd

    mesh = fd.UnitSquareMesh(n, n)
    V = fd.VectorFunctionSpace(mesh, "CG", 2)

    a = IMEX_firedrake_mesh(V, val=v1)
    b = IMEX_firedrake_mesh(V, val=v2)

    c = a + b

    assert np.allclose(c.impl.dat._numpy_data, v1 + v2)
    assert np.allclose(c.expl.dat._numpy_data, v1 + v2)
    assert np.allclose(a.impl.dat._numpy_data, v1)
    assert np.allclose(b.impl.dat._numpy_data, v2)
    assert np.allclose(a.expl.dat._numpy_data, v1)
    assert np.allclose(b.expl.dat._numpy_data, v2)


@pytest.mark.firedrake
def test_subtraction_rhs(n=3, v1=1, v2=2):
    from pySDC.implementations.datatype_classes.firedrake_mesh import IMEX_firedrake_mesh
    import numpy as np
    import firedrake as fd

    mesh = fd.UnitSquareMesh(n, n)
    V = fd.VectorFunctionSpace(mesh, "CG", 2)

    a = IMEX_firedrake_mesh(V, val=v1)
    b = IMEX_firedrake_mesh(V, val=v2)

    c = a - b

    assert np.allclose(c.impl.dat._numpy_data, v1 - v2)
    assert np.allclose(c.expl.dat._numpy_data, v1 - v2)
    assert np.allclose(a.impl.dat._numpy_data, v1)
    assert np.allclose(b.impl.dat._numpy_data, v2)
    assert np.allclose(a.expl.dat._numpy_data, v1)
    assert np.allclose(b.expl.dat._numpy_data, v2)


@pytest.mark.firedrake
def test_rmul_rhs(n=3, v1=1, v2=2):
    from pySDC.implementations.datatype_classes.firedrake_mesh import IMEX_firedrake_mesh
    import numpy as np
    import firedrake as fd

    mesh = fd.UnitSquareMesh(n, n)
    V = fd.VectorFunctionSpace(mesh, "CG", 2)

    a = IMEX_firedrake_mesh(V, val=v1)

    b = v2 * a

    assert np.allclose(a.impl.dat._numpy_data, v1)
    assert np.allclose(b.impl.dat._numpy_data, v2 * v1)
    assert np.allclose(a.expl.dat._numpy_data, v1)
    assert np.allclose(b.expl.dat._numpy_data, v2 * v1)


if __name__ == '__main__':
    test_addition()
