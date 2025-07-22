import numpy as np
import pytest

from pySDC.core.sweeper import Sweeper as Sweeper

node_types = ['EQUID', 'LEGENDRE'] + [f'CHEBY-{i}' for i in [1, 2, 3, 4]]
quad_types = ['GAUSS', 'LOBATTO', 'RADAU-RIGHT', 'RADAU-LEFT']
num_nodes = [2, 3, 4, 5]


@pytest.mark.base
@pytest.mark.parametrize("node_type", node_types)
@pytest.mark.parametrize("quad_type", quad_types)
@pytest.mark.parametrize("M", num_nodes)
def test_MIN_SR(node_type, quad_type, M):
    params = {'num_nodes': M, 'quad_type': quad_type, 'node_type': node_type}
    sweeper = Sweeper(params, None)
    Q = sweeper.coll.Qmat[1:, 1:]

    # Check non-stiff limit
    QDelta = sweeper.get_Qdelta_implicit('MIN-SR-NS')[1:, 1:]
    assert np.all(np.diag(np.diag(QDelta)) == QDelta), "QDelta not diagonal"
    K = Q - QDelta
    Km = np.linalg.matrix_power(K, M)
    nilpotency = np.linalg.norm(Km, ord=np.inf)
    assert nilpotency < 1e-10, "Q-QDelta not nilpotent " f"(M={M}, norm={nilpotency})"

    # Check stiff limit
    QDelta = sweeper.get_Qdelta_implicit('MIN-SR-S')[1:, 1:]
    assert np.all(np.diag(np.diag(QDelta)) == QDelta), "QDelta not diagonal"

    if params['quad_type'] in ['LOBATTO', 'RADAU-LEFT']:
        QDelta = np.diag(1 / np.diag(QDelta[1:, 1:]))
        Q = Q[1:, 1:]
    else:
        QDelta = np.diag(1 / np.diag(QDelta))

    K = np.eye(Q.shape[0]) - QDelta @ Q
    Km = np.linalg.matrix_power(K, M)
    nilpotency = np.linalg.norm(Km, ord=np.inf)
    assert nilpotency < 1e-10, "I-QDelta^{-1}Q not nilpotent " f"(M={M}, norm={nilpotency})"


@pytest.mark.base
@pytest.mark.parametrize("node_type", node_types)
@pytest.mark.parametrize("quad_type", quad_types)
@pytest.mark.parametrize("M", num_nodes)
def test_MIN_SR_FLEX(node_type, quad_type, M):
    params = {'num_nodes': M, 'quad_type': quad_type, 'node_type': node_type}
    sweeper = Sweeper(params, None)

    start_idx = 1
    for i in range(M):
        if sweeper.coll.nodes[i] == 0:
            start_idx += 1
        else:
            break

    Q = sweeper.coll.Qmat[start_idx:, start_idx:]

    QDelta = [sweeper.get_Qdelta_implicit('MIN-SR-FLEX', k=i + 1)[start_idx:, start_idx:] for i in range(M)]
    for QD in QDelta:
        assert np.all(np.diag(np.diag(QD)) == QD), "QDelta not diagonal"

    I = np.eye(M + 1 - start_idx)
    K = np.eye(M + 1 - start_idx)
    for QD in QDelta:
        K = (I - np.linalg.inv(QD) @ Q) @ K

    nilpotency = np.linalg.norm(K, ord=np.inf)
    assert (
        nilpotency < 1e-10
    ), f"Applying FLEX preconditioner does not give nilpotent SDC iteration matrix after {M} iterations! (M={M}, norm={nilpotency})"


@pytest.mark.base
@pytest.mark.parametrize('imex', [True, False])
@pytest.mark.parametrize('num_nodes', num_nodes)
def test_FLEX_preconditioner_in_sweepers(imex, num_nodes, MPI=False):
    from pySDC.core.level import Level

    if imex:
        from pySDC.implementations.problem_classes.TestEquation_0D import test_equation_IMEX as problem_class

        if MPI:
            from pySDC.implementations.sweeper_classes.imex_1st_order_MPI import imex_1st_order_MPI as sweeper_class
        else:
            from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order as sweeper_class
    else:
        from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d as problem_class

        if MPI:
            from pySDC.implementations.sweeper_classes.generic_implicit_MPI import generic_implicit_MPI as sweeper_class
        else:
            from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class

    sweeper_params = {'quad_type': 'RADAU-RIGHT', 'num_nodes': num_nodes, 'QI': 'MIN-SR-FLEX', 'QE': 'PIC'}
    if MPI:
        from mpi4py import MPI

        sweeper_params['comm'] = MPI.COMM_WORLD
    level_params = {'nsweeps': num_nodes, 'dt': 1}

    lvl = Level(problem_class, {}, sweeper_class, sweeper_params, level_params, 0)

    lvl.status.unlocked = True
    lvl.u[0] = lvl.prob.u_exact(0)
    lvl.status.time = 0

    sweep = lvl.sweep
    sweep.predict()

    for k in range(1, level_params['nsweeps'] + 1):
        lvl.status.sweep = k
        sweep.update_nodes()
        assert np.allclose(
            sweep.QI, sweep.get_Qdelta_implicit(sweeper_params['QI'], k)
        ), f'Got incorrect FLEX preconditioner in sweep {k}'


@pytest.mark.mpi4py
@pytest.mark.parametrize('imex', [True, False])
@pytest.mark.mpi(ranks=[3])
def test_FLEX_preconditioner_in_MPI_sweepers(mpi_ranks, imex):
    from mpi4py import MPI

    test_FLEX_preconditioner_in_sweepers(imex, num_nodes=MPI.COMM_WORLD.size, MPI=True)


@pytest.mark.base
@pytest.mark.parametrize("node_type", node_types)
@pytest.mark.parametrize("quad_type", quad_types)
@pytest.mark.parametrize("M", num_nodes)
def test_LU(node_type, quad_type, M):
    if M > 3 and node_type == 'EQUID' and quad_type in ['GAUSS', 'RADAU-RIGHT']:
        # Edge case for some specific equidistant nodes
        # TODO : still need to be understood ...
        return

    params = {'num_nodes': M, 'quad_type': quad_type, 'node_type': node_type}
    sweeper = Sweeper(params, None)
    Q = sweeper.coll.Qmat[1:, 1:]

    # Check nilpotency
    QDelta = sweeper.get_Qdelta_implicit('LU')[1:, 1:]

    if params['quad_type'] in ['LOBATTO', 'RADAU-LEFT']:
        QDelta = QDelta[1:, 1:]
        Q = Q[1:, 1:]

    K = np.eye(Q.shape[0]) - np.linalg.solve(QDelta, Q)
    Km = np.linalg.matrix_power(K, M)
    nilpotency = np.linalg.norm(Km, ord=np.inf)
    assert nilpotency < 1e-14, "I-QDelta^{-1}Q not nilpotent " f"(M={M}, norm={nilpotency})"


@pytest.mark.base
@pytest.mark.parametrize("node_type", node_types)
@pytest.mark.parametrize("quad_type", quad_types)
@pytest.mark.parametrize("M", num_nodes)
def test_Qpar(node_type, quad_type, M):
    params = {'num_nodes': M, 'quad_type': quad_type, 'node_type': node_type}
    sweeper = Sweeper(params, None)
    Q = sweeper.coll.Qmat[1:, 1:]

    QDelta = sweeper.get_Qdelta_implicit('Qpar')[1:, 1:]
    assert np.all(np.diag(np.diag(QDelta)) == QDelta), "no diagonal QDelta"
    assert np.all(np.diag(QDelta) == np.diag(Q)), "not the diagonal Q coefficients"


@pytest.mark.base
@pytest.mark.parametrize("node_type", node_types)
@pytest.mark.parametrize("quad_type", quad_types)
@pytest.mark.parametrize("M", num_nodes)
def test_IE(node_type, quad_type, M):
    params = {'num_nodes': M, 'quad_type': quad_type, 'node_type': node_type}
    sweeper = Sweeper(params, None)

    QDelta = sweeper.get_Qdelta_implicit('IE')[1:, 1:]
    for i in range(M):
        assert np.all(QDelta[i, : i + 1] == QDelta[-1, : i + 1]), "not the same coefficients in columns"
    assert np.all(np.cumsum(QDelta[-1] == sweeper.coll.nodes)), "last line cumsum not equal to nodes"


@pytest.mark.base
@pytest.mark.parametrize("node_type", node_types)
@pytest.mark.parametrize("quad_type", quad_types)
@pytest.mark.parametrize("M", num_nodes)
def test_IEpar(node_type, quad_type, M):
    params = {'num_nodes': M, 'quad_type': quad_type, 'node_type': node_type}
    sweeper = Sweeper(params, None)

    QDelta = sweeper.get_Qdelta_implicit('IEpar')[1:, 1:]
    assert np.all(np.diag(np.diag(QDelta)) == QDelta), "no diagonal QDelta"
    assert np.all(np.cumsum(np.diag(QDelta) == sweeper.coll.nodes)), "diagonal cumsum not equal to nodes"


@pytest.mark.base
@pytest.mark.parametrize("node_type", node_types)
@pytest.mark.parametrize("quad_type", quad_types)
@pytest.mark.parametrize("M", num_nodes)
def test_PIC(node_type, quad_type, M):
    params = {'num_nodes': M, 'quad_type': quad_type, 'node_type': node_type}
    sweeper = Sweeper(params, None)

    QDelta = sweeper.get_Qdelta_implicit('PIC')[1:, 1:]
    assert np.all(QDelta == 0), "not a null matrix"


if __name__ == '__main__':
    test_MIN_SR_FLEX('LEGENDRE', 'LOBATTO', 4)

    test_MIN_SR('LEGENDRE', 'RADAU-RIGHT', 4)
    test_MIN_SR('EQUID', 'LOBATTO', 5)

    test_LU('LEGENDRE', 'RADAU-RIGHT', 4)
    test_LU('EQUID', 'LOBATTO', 5)

    test_FLEX_preconditioner_in_sweepers(True)
