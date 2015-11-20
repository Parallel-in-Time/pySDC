import pySDC.Collocation
from pySDC.CollocationClasses import *
import numpy as np
import pytest

# py.test excludes classes with a constructor by default, so define parameter here
t_start = 0.0
t_end   = 1.0
#classes = [ ["CollGaussLegendre", 2, 2]]
classes = [ ["CollGaussLegendre", 2, 12], ["CollGaussLobatto", 2, 12], ["CollGaussRadau_Right", 2, 12] ]

class TestCollocation:

  # TEST 1:
  # Check that the quadrature rule integrates polynomials up to order p-1 exactly
  # -----------------------------------------------------------------------------
  def test_1(self):
    for type in classes:
      for M in range(type[1],type[2]+1):
        coll = getattr(pySDC.CollocationClasses, type[0])(M, t_start, t_end)
        
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
        assert abs(int_ex - int_coll) < 1e-10, "For node type " + type[0] + ", failed to integrate polynomial of degree " + str(coll.order-1) + " exactly. Error: %5.3e" % abs(int_ex - int_coll)


  # TEST 2:
  # Check that the Qmat entries are equal to the sum of Smat entries
  # ----------------------------------------------------------------
  def test_2(self):
    for type in classes:
      for M in range(type[1],type[2]+1):
        coll = getattr(pySDC.CollocationClasses, type[0])(M, t_start, t_end)
        Q = coll.Qmat[1:,1:]
        S = coll.Smat[1:,1:]
        assert np.shape(Q) == np.shape(S), "For node type " + type[0] + ", Qmat and Smat have different shape"
        shape = np.shape(Q)
        assert shape[0] == shape[1], "For node type " + type[0] + ", Qmat / Smat are not quadratic"
        SSum = np.cumsum(S[:,:],axis=0)
        for i in range(0,M):
          assert np.linalg.norm( Q[i,:] - SSum[i,:] ) < 1e-15, "For node type " + type[0] + ", Qmat and Smat did not satisfy the expected summation property."

  # TEST 3:
  # Check that the partial quadrature rules from Qmat entries have order equal to number of nodes M
  # -----------------------------------------------------------------------------------------------
  def test_3(self):
    for type in classes:
      for M in range(type[1],type[2]+1):
        coll = getattr(pySDC.CollocationClasses, type[0])(M, t_start, t_end)
        Q = coll.Qmat[1:,1:]
        # as in TEST 1, create and integrate a polynomial with random coefficients, but now of degree M-1
        poly_coeff = np.random.rand(M-1)
        poly_vals  = np.polyval(poly_coeff, coll.nodes)
        poly_int_coeff = np.polyint(poly_coeff)
        for i in range(0,M):
            int_ex = np.polyval(poly_int_coeff, coll.nodes[i]) - np.polyval(poly_int_coeff, t_start)
            int_coll = np.dot(poly_vals, Q[i,:])
            assert abs(int_ex - int_coll)<1e-12, "For node type " + type[0] + ", partial quadrature from Qmat rule failed to integrate polynomial of degree M-1 exactly for M = " + str(M)

  def test_4(self):
    for type in classes:
      for M in range(type[1],type[2]+1):
        coll = getattr(pySDC.CollocationClasses, type[0])(M, t_start, t_end)
        S = coll.Smat[1:,1:]
        # as in TEST 1, create and integrate a polynomial with random coefficients, but now of degree M-1
        poly_coeff = np.random.rand(M-1)
        poly_vals  = np.polyval(poly_coeff, coll.nodes)
        poly_int_coeff = np.polyint(poly_coeff)
        for i in range(1,M):
            int_ex = np.polyval(poly_int_coeff, coll.nodes[i]) - np.polyval(poly_int_coeff, coll.nodes[i-1])
            int_coll = np.dot(poly_vals, S[i,:])
            assert abs(int_ex - int_coll)<1e-12, "For node type " + type[0] + ", partial quadrature rule from Smat failed to integrate polynomial of degree M-1 exactly for M = " + str(M)
