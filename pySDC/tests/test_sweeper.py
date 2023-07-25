import numpy as np
import pytest

from pySDC.core.Sweeper import sweeper as Sweeper

node_types = ['EQUID', 'LEGENDRE'] + [f'CHEBY-{i}' for i in [1, 2, 3, 4]]
quad_types = ['GAUSS', 'LOBATTO', 'RADAU-RIGHT', 'RADAU-LEFT']


@pytest.mark.base
@pytest.mark.parametrize("node_type", node_types)
@pytest.mark.parametrize("quad_type", quad_types)
def test_MIN_SR(node_type, quad_type):
    for M in [2, 3, 4, 5]:
        params = {'num_nodes': M, 'quad_type': quad_type, 'node_type': node_type}

        sweeper = Sweeper(params)
        Q = sweeper.coll.Qmat[1:, 1:]

        # Check non-stiff limit
        QDelta = sweeper.get_Qdelta_implicit(sweeper.coll, 'MIN-SR-NS')[1:, 1:]
        assert np.all(np.diag(np.diag(QDelta)) == QDelta), "no diagonal QDelta"
        K = Q - QDelta
        Km = np.linalg.matrix_power(K, M)
        nilpotency = np.linalg.norm(Km, ord=np.inf)
        assert nilpotency < 1e-10, "Q-QDelta not nilpotent " f"(M={M}, norm={nilpotency})"

        # Check stiff limit
        QDelta = sweeper.get_Qdelta_implicit(sweeper.coll, 'MIN-SR-S')[1:, 1:]
        assert np.all(np.diag(np.diag(QDelta)) == QDelta), "no diagonal QDelta"

        if params['quad_type'] in ['LOBATTO', 'RADAU-LEFT']:
            QDelta = np.diag(1 / np.diag(QDelta[1:, 1:]))
            Q = Q[1:, 1:]
        else:
            QDelta = np.diag(1 / np.diag(QDelta))

        K = np.eye(Q.shape[0]) - QDelta @ Q
        Km = np.linalg.matrix_power(K, M)
        nilpotency = np.linalg.norm(Km, ord=np.inf)
        assert nilpotency < 1e-10, "I-QDelta^{-1}Q not nilpotent " f"(M={M}, norm={nilpotency})"


if __name__ == '__main__':
    test_MIN_SR('LEGENDRE', 'RADAU-RIGHT')
    test_MIN_SR('EQUID', 'LOBATTO')
