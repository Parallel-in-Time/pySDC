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

    assert np.allclose(c.dat.data_ro, v1 + v2)
    assert np.allclose(a.dat.data_ro, v1)
    assert np.allclose(b.dat.data_ro, v2)


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

    assert np.allclose(c.dat.data_ro, v1 - v2)
    assert np.allclose(a.dat.data_ro, v1)
    assert np.allclose(b.dat.data_ro, v2)


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

    assert np.allclose(b.dat.data_ro, v1 * v2)
    assert np.allclose(a.dat.data_ro, v1)

    with pytest.raises(DataError):
        'Dat kölsche Dom' * b


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
    assert np.allclose(a.dat.data_ro, v1)


@pytest.mark.firedrake
def test_dual_data(n=3):
    """Dual data stay dual through copies and arithmetic, and cannot be mixed with primal data."""
    from pySDC.core.errors import DataError
    from pySDC.implementations.datatype_classes.firedrake_mesh import (
        firedrake_mesh,
        firedrake_dual_mesh,
        IMEX_firedrake_dual_mesh,
    )
    import numpy as np
    import firedrake as fd

    V = fd.FunctionSpace(fd.UnitSquareMesh(n, n), "CG", 2)

    a = firedrake_dual_mesh(V, val=1.0)
    b = firedrake_mesh(fd.assemble(fd.TestFunction(V) * fd.dx))
    assert isinstance(a.functionspace, fd.Cofunction) and a.is_dual and b.is_dual

    for c in [firedrake_mesh(a), firedrake_dual_mesh(a), a + b, a - b, 2.0 * a]:
        assert c.is_dual
    assert np.allclose((a + 2.0 * b).dat.data_ro, 1 + 2 * b.dat.data_ro)

    u = firedrake_mesh(V, val=1.0)
    assert not u.is_dual
    for operation in [lambda: u + a, lambda: a - u]:
        with pytest.raises(DataError):
            operation()

    f = IMEX_firedrake_dual_mesh(V)
    assert f.impl.is_dual and f.expl.is_dual and (f.impl + f.expl).is_dual


@pytest.mark.firedrake
def test_dual_norm():
    """
    The norm of dual data is that of the function they represent, so the load vector of a function has the norm of its
    L2 projection on every mesh, while the norm of the entries shrinks with the mesh width.
    """
    from pySDC.implementations.datatype_classes.firedrake_mesh import firedrake_mesh
    import numpy as np
    import firedrake as fd

    norms, entries = [], []
    for n in [4, 16]:
        V = fd.FunctionSpace(fd.UnitSquareMesh(n, n), "CG", 2)
        x, y = fd.SpatialCoordinate(V.mesh())
        expr = fd.sin(np.pi * x) * y
        r = firedrake_mesh(fd.assemble(expr * fd.TestFunction(V) * fd.dx))

        assert np.isclose(abs(r), fd.norm(fd.project(expr, V)), rtol=1e-6)
        norms.append(abs(r))
        entries.append(np.linalg.norm(r.dat.data_ro))

    exact = (1 / 6) ** 0.5  # L2 norm of sin(pi x) y on the unit square
    assert np.allclose(norms, exact, rtol=1e-3), norms
    assert entries[1] < entries[0] / 3, entries


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

    assert np.allclose(c.impl.dat.data_ro, v1 + v2)
    assert np.allclose(c.expl.dat.data_ro, v1 + v2)
    assert np.allclose(a.impl.dat.data_ro, v1)
    assert np.allclose(b.impl.dat.data_ro, v2)
    assert np.allclose(a.expl.dat.data_ro, v1)
    assert np.allclose(b.expl.dat.data_ro, v2)


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

    assert np.allclose(c.impl.dat.data_ro, v1 - v2)
    assert np.allclose(c.expl.dat.data_ro, v1 - v2)
    assert np.allclose(a.impl.dat.data_ro, v1)
    assert np.allclose(b.impl.dat.data_ro, v2)
    assert np.allclose(a.expl.dat.data_ro, v1)
    assert np.allclose(b.expl.dat.data_ro, v2)


@pytest.mark.firedrake
def test_rmul_rhs(n=3, v1=1, v2=2):
    from pySDC.implementations.datatype_classes.firedrake_mesh import IMEX_firedrake_mesh
    import numpy as np
    import firedrake as fd

    mesh = fd.UnitSquareMesh(n, n)
    V = fd.VectorFunctionSpace(mesh, "CG", 2)

    a = IMEX_firedrake_mesh(V, val=v1)

    b = v2 * a

    assert np.allclose(a.impl.dat.data_ro, v1)
    assert np.allclose(b.impl.dat.data_ro, v2 * v1)
    assert np.allclose(a.expl.dat.data_ro, v1)
    assert np.allclose(b.expl.dat.data_ro, v2 * v1)


def _test_p2p_communication(comm, u):
    import numpy as np

    assert comm.size == 2
    if comm.rank == 0:
        u.assign(3.14)
        req = u.isend(dest=1, comm=comm, tag=0)
    elif comm.rank == 1:
        assert not np.allclose(u.dat.data_ro, 3.14)
        req = u.irecv(source=0, comm=comm, tag=0)
    req.wait()
    assert np.allclose(u.dat.data_ro, 3.14)


def _test_bcast(comm, u):
    import numpy as np

    if comm.rank == 0:
        u.assign(3.14)
    else:
        assert not np.allclose(u.dat.data_ro, 3.14)
    u.bcast(root=0, comm=comm)
    assert np.allclose(u.dat.data_ro, 3.14)


@pytest.mark.firedrake
@pytest.mark.parallel(2)
@pytest.mark.parametrize('pattern', ['p2p', 'bcast'])
def test_communication(pattern, n=2):
    import firedrake as fd
    from pySDC.helpers.firedrake_ensemble_communicator import FiredrakeEnsembleCommunicator
    from pySDC.implementations.datatype_classes.firedrake_mesh import firedrake_mesh

    ensemble_comm = FiredrakeEnsembleCommunicator(fd.COMM_WORLD, 1)

    mesh = fd.UnitSquareMesh(n, n, comm=ensemble_comm.space_comm)
    V = fd.VectorFunctionSpace(mesh, "CG", 2)

    u = firedrake_mesh(V)

    if pattern == 'p2p':
        _test_p2p_communication(ensemble_comm, u)
    elif pattern == 'bcast':
        _test_bcast(ensemble_comm, u)
    else:
        raise NotImplementedError


@pytest.mark.firedrake
@pytest.mark.parallel(2)
def test_communication_of_mixed_functions(n=2):
    """Every subfunction of a function on a mixed space arrives, and sending needs no tag, like in mpi4py."""
    import firedrake as fd
    import numpy as np
    from pySDC.helpers.firedrake_ensemble_communicator import FiredrakeEnsembleCommunicator

    comm = FiredrakeEnsembleCommunicator(fd.COMM_WORLD, 1)
    mesh = fd.UnitSquareMesh(n, n, comm=comm.space_comm)
    W = fd.FunctionSpace(mesh, 'CG', 1) * fd.FunctionSpace(mesh, 'DG', 0)
    u = fd.Function(W)

    if comm.rank == 0:
        for i, sub in enumerate(u.subfunctions):
            sub.assign(i + 1)
        req = comm.Isend(u, dest=1)
    else:
        req = comm.Irecv(u, source=0, tag=0)
    req.Wait()

    for i, sub in enumerate(u.subfunctions):
        assert np.allclose(sub.dat.data_ro, i + 1), f'Subfunction {i} did not arrive'


@pytest.mark.firedrake
@pytest.mark.parallel(2)
def test_free_releases_only_split_communicators():
    import firedrake as fd
    from mpi4py import MPI
    from pySDC.helpers.firedrake_ensemble_communicator import FiredrakeEnsembleCommunicator

    comm = FiredrakeEnsembleCommunicator(fd.COMM_WORLD, 1)
    split = comm.Split(0)
    split.Free()
    assert split.comm_wold == MPI.COMM_NULL

    comm.Free()
    assert fd.COMM_WORLD != MPI.COMM_NULL
