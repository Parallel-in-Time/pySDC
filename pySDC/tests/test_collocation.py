import pytest
import numpy as np

from pySDC.core.Collocation import CollBase

t_start = np.random.rand(1) * 0.2
t_end = 0.8 + np.random.rand(1) * 0.2

tolQuad = 1e-13

node_types = ['EQUID', 'LEGENDRE']
quad_types = ['GAUSS', 'LOBATTO', 'RADAU-RIGHT', 'RADAU-LEFT']

tolQuad = 1e-13

@pytest.mark.base
@pytest.mark.parametrize("node_type", node_types)
@pytest.mark.parametrize("quad_type", quad_types)
def test_canintegratepolynomials(node_type, quad_type):

    for M in range(2, 13):

        coll = CollBase(M, t_start, t_end, node_type=node_type, quad_type=quad_type)

        # some basic consistency tests
        assert np.size(coll.nodes) == np.size(coll.weights), (
            "For node type " + coll.__class__.__name__ + ", number of entries in nodes and weights is different"
        )
        assert np.size(coll.nodes) == M, (
            "For node type "
            + coll.__class__.__name__
            + ", requesting M nodes did not produce M entries in nodes and weights"
        )

        # generate random set of polynomial coefficients
        poly_coeff = np.random.rand(coll.order - 1)
        # evaluate polynomial at collocation nodes
        poly_vals = np.polyval(poly_coeff, coll.nodes)
        # use python's polyint function to compute anti-derivative of polynomial
        poly_int_coeff = np.polyint(poly_coeff)
        # Compute integral from 0.0 to 1.0
        int_ex = np.polyval(poly_int_coeff, t_end) - np.polyval(poly_int_coeff, t_start)
        # use quadrature rule to compute integral
        int_coll = coll.evaluate(coll.weights, poly_vals)
        # For large values of M, substantial differences from different round of error have to be considered
        assert abs(int_ex - int_coll) < tolQuad, (
            "For node type "
            + coll.__class__.__name__
            + ", failed to integrate polynomial of degree "
            + str(coll.order - 1)
            + " exactly. Error: %5.3e" % abs(int_ex - int_coll)
        )


@pytest.mark.base
@pytest.mark.parametrize("node_type", node_types)
@pytest.mark.parametrize("quad_type", quad_types)
def test_relateQandSmat(node_type, quad_type):
    for M in range(2, 13):
        coll = CollBase(M, t_start, t_end, node_type=node_type, quad_type=quad_type)
        Q = coll.Qmat[1:, 1:]
        S = coll.Smat[1:, 1:]
        assert np.shape(Q) == np.shape(S), (
            "For node type " + coll.__class__.__name__ + ", Qmat and Smat have different shape"
        )
        shape = np.shape(Q)
        assert shape[0] == shape[1], "For node type " + coll.__class__.__name__ + ", Qmat / Smat are not quadratic"
        SSum = np.cumsum(S[:, :], axis=0)
        for i in range(0, M):
            assert np.linalg.norm(Q[i, :] - SSum[i, :]) < 1e-15, (
                "For node type "
                + coll.__class__.__name__
                + ", Qmat and Smat did not satisfy the expected summation property."
            )


@pytest.mark.base
@pytest.mark.parametrize("node_type", node_types)
@pytest.mark.parametrize("quad_type", quad_types)
def test_partialquadraturewithQ(node_type, quad_type):
    for M in range(2, 13):
        coll = CollBase(M, t_start, t_end, node_type=node_type, quad_type=quad_type)
        Q = coll.Qmat[1:, 1:]
        # as in TEST 1, create and integrate a polynomial with random coefficients, but now of degree M-1 (or less for splines)
        degree = min(coll.order, M - 1)
        poly_coeff = np.random.rand(degree)
        poly_vals = np.polyval(poly_coeff, coll.nodes)
        poly_int_coeff = np.polyint(poly_coeff)
        for i in range(0, M):
            int_ex = np.polyval(poly_int_coeff, coll.nodes[i]) - np.polyval(poly_int_coeff, t_start)
            int_coll = np.dot(poly_vals, Q[i, :])
            assert abs(int_ex - int_coll) < tolQuad, (
                "For node type "
                + coll.__class__.__name__
                + ", partial quadrature from Qmat rule failed to integrate polynomial of degree M-1 exactly for M = "
                + str(M)
            )


@pytest.mark.base
@pytest.mark.parametrize("node_type", node_types)
@pytest.mark.parametrize("quad_type", quad_types)
def test_partialquadraturewithS(node_type, quad_type):
    for M in range(2, 13):
        coll = CollBase(M, t_start, t_end, node_type=node_type, quad_type=quad_type)
        S = coll.Smat[1:, 1:]
        # as in TEST 1, create and integrate a polynomial with random coefficients, but now of degree M-1 (or less for splines)
        degree = min(coll.order, M - 1)
        poly_coeff = np.random.rand(degree)
        poly_vals = np.polyval(poly_coeff, coll.nodes)
        poly_int_coeff = np.polyint(poly_coeff)
        for i in range(1, M):
            int_ex = np.polyval(poly_int_coeff, coll.nodes[i]) - np.polyval(poly_int_coeff, coll.nodes[i - 1])
            int_coll = np.dot(poly_vals, S[i, :])
            assert abs(int_ex - int_coll) < tolQuad, (
                "For node type "
                + coll.__class__.__name__
                + ", partial quadrature rule from Smat failed to integrate polynomial of degree M-1 exactly for M = "
                + str(M)
            )
