import nose
import numpy as np

from pySDC.core.Collocation import CollBase
from pySDC.tests.test_helpers import get_derived_from_in_package

classes = []
t_start = None
t_end = None

def setup():
    global classes, t_start, t_end

    # generate random boundaries for the time slice with 0.0 <= t_start < 0.2 and 0.8 <= t_end < 1.0
    t_start = np.random.rand(1) * 0.2
    t_end = 0.8 + np.random.rand(1) * 0.2
    classes = get_derived_from_in_package(CollBase, 'pySDC/implementations/collocation_classes')


# TEST 1:
  # Check that the quadrature rule integrates polynomials up to order p-1 exactly
  # -----------------------------------------------------------------------------
@nose.tools.with_setup(setup)
def test_canintegratepolynomials():
    for collclass in classes:
        yield check_canintegratepolynomials, collclass, t_start, t_end

def check_canintegratepolynomials(collclass,t_start,t_end):

    for M in range(2,13):
        coll = collclass(M, t_start, t_end)

        # some basic consistency tests
        assert np.size(coll.nodes)==np.size(coll.weights), "For node type " + type[0] + ", number of entries in nodes and weights is different"
        assert np.size(coll.nodes)==M, "For node type " + type[0] + ", requesting M nodes did not produce M entries in nodes and weights"

        # generate random set of polynomial coefficients
        poly_coeff = np.random.rand(coll.order-1)
        # evaluate polynomial at collocation nodes
        poly_vals  = np.polyval(poly_coeff, coll.nodes)
        # use python's polyint function to compute anti-derivative of polynomial
        poly_int_coeff = np.polyint(poly_coeff)
        # Compute integral from 0.0 to 1.0
        int_ex = np.polyval(poly_int_coeff, t_end) - np.polyval(poly_int_coeff, t_start)
        # use quadrature rule to compute integral
        int_coll = coll.evaluate(coll.weights, poly_vals)
        # For large values of M, substantial differences from different round of error have to be considered
        assert abs(int_ex - int_coll) < 1e-13, "For node type " + coll.__class__.__name__ + ", failed to integrate polynomial of degree " + str(coll.order-1) + " exactly. Error: %5.3e" % abs(int_ex - int_coll)


# TEST 2:
# Check that the Qmat entries are equal to the sum of Smat entries
# ----------------------------------------------------------------
@nose.tools.with_setup(setup)
def test_relateQandSmat():
    for collclass in classes:
        yield check_relateQandSmat, collclass, t_start, t_end


def check_relateQandSmat(collclass,t_start,t_end):
    for M in range(2, 13):
        coll = collclass(M, t_start, t_end)
        Q = coll.Qmat[1:,1:]
        S = coll.Smat[1:,1:]
        assert np.shape(Q) == np.shape(S), "For node type " + type[0] + ", Qmat and Smat have different shape"
        shape = np.shape(Q)
        assert shape[0] == shape[1], "For node type " + type[0] + ", Qmat / Smat are not quadratic"
        SSum = np.cumsum(S[:,:],axis=0)
        for i in range(0,M):
          assert np.linalg.norm( Q[i,:] - SSum[i,:] ) < 1e-15, "For node type " + coll.__class__.__name__ + ", Qmat and Smat did not satisfy the expected summation property."


# TEST 3:
# Check that the partial quadrature rules from Qmat entries have order equal to number of nodes M
# -----------------------------------------------------------------------------------------------
@nose.tools.with_setup(setup)
def test_partialquadrature():
    for collclass in classes:
        yield check_partialquadraturewithQ, collclass, t_start, t_end

def check_partialquadraturewithQ(collclass, t_start, t_end):
    for M in range(2, 13):
        coll = collclass(M, t_start, t_end)
        Q = coll.Qmat[1:,1:]
        # as in TEST 1, create and integrate a polynomial with random coefficients, but now of degree M-1 (or less for splines)
        degree = min(coll.order,M-1)
        poly_coeff = np.random.rand(degree)
        poly_vals  = np.polyval(poly_coeff, coll.nodes)
        poly_int_coeff = np.polyint(poly_coeff)
        for i in range(0,M):
            int_ex = np.polyval(poly_int_coeff, coll.nodes[i]) - np.polyval(poly_int_coeff, t_start)
            int_coll = np.dot(poly_vals, Q[i,:])
            assert abs(int_ex - int_coll)<1e-12, "For node type " + coll.__class__.__name__ + ", partial quadrature from Qmat rule failed to integrate polynomial of degree M-1 exactly for M = " + str(M)

# TEST 3:
# Check that the partial quadrature rules from Smat entries have order equal to number of nodes M
# -----------------------------------------------------------------------------------------------
@nose.tools.with_setup(setup)
def test_partialquadraturewithS():
    for collclass in classes:
        yield check_partialquadraturewithS, collclass, t_start, t_end

def check_partialquadraturewithS(collclass, t_start, t_end):
    for M in range(2, 13):
        coll = collclass(M, t_start, t_end)
        S = coll.Smat[1:,1:]
        # as in TEST 1, create and integrate a polynomial with random coefficients, but now of degree M-1 (or less for splines)
        degree = min(coll.order, M - 1)
        poly_coeff = np.random.rand(degree)
        poly_vals  = np.polyval(poly_coeff, coll.nodes)
        poly_int_coeff = np.polyint(poly_coeff)
        for i in range(1,M):
            int_ex = np.polyval(poly_int_coeff, coll.nodes[i]) - np.polyval(poly_int_coeff, coll.nodes[i-1])
            int_coll = np.dot(poly_vals, S[i,:])
            assert abs(int_ex - int_coll)<1e-12, "For node type " + coll.__class__.__name__ + ", partial quadrature rule from Smat failed to integrate polynomial of degree M-1 exactly for M = " + str(M)
