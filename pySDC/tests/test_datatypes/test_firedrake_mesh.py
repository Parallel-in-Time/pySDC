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


def _test_p2p_communication(comm, u):
    import numpy as np

    assert comm.size == 2
    if comm.rank == 0:
        u.assign(3.14)
        req = u.isend(dest=1, comm=comm, tag=0)
    elif comm.rank == 1:
        assert not np.allclose(u.dat._numpy_data, 3.14)
        req = u.irecv(source=0, comm=comm, tag=0)
    req.wait()
    assert np.allclose(u.dat._numpy_data, 3.14)


def _test_bcast(comm, u):
    import numpy as np

    if comm.rank == 0:
        u.assign(3.14)
    else:
        assert not np.allclose(u.dat._numpy_data, 3.14)
    u.bcast(root=0, comm=comm)
    assert np.allclose(u.dat._numpy_data, 3.14)


@pytest.mark.firedrake
@pytest.mark.parametrize('pattern', ['p2p', 'bcast'])
def test_communication(pattern, n=2, submit=True):
    if submit:
        import os
        import subprocess

        my_env = os.environ.copy()
        my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'
        cwd = '.'
        num_procs = 2
        cmd = f'mpiexec -np {num_procs} python {__file__} --pattern {pattern}'.split()

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env, cwd=cwd)
        p.wait()
        for line in p.stdout:
            print(line)
        for line in p.stderr:
            print(line)
        assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (
            p.returncode,
            num_procs,
        )

    else:
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


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        '--pattern',
        help="pattern for parallel tests",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    if args.pattern:
        test_communication(pattern=args.pattern, submit=False)
