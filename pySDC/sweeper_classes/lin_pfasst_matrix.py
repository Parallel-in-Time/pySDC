# coding=utf-8
import numpy as np
import scipy as sp
import scipy.sparse as sprs
import scipy.sparse.linalg as spla
import scipy.linalg as la
import scipy.interpolate as intpl
from pySDC.Sweeper import sweeper
from pySDC.tools.transfer_tools import to_sparse, to_dense
from pySDC.tools.matrix_method_tools import *

class IterativeSolver(sweeper):
    """ The basic Iterative Solver class,
        several steps of the iterative solver are called sweeps"""
    def __init__(self, P, M, c, sparse_format="array"):
        assert P.shape == M.shape and c.shape[0] == M.shape[0], \
            "Matrix P and matrix M don't fit.\n \tM_shape:" + str(M.shape) + " P_shape:" \
            + str(P.shape) + " c_shape:" + str(c.shape)
        self.sparse_format = sparse_format
        self.P = to_sparse(P, sparse_format)
        self.M = to_sparse(M, sparse_format)
        self.c = to_sparse(c, sparse_format)

    def invert_P(self):
        return la.inv(self.P)

    @property
    def P_inv(self):
        if self.sparse_format is "array":
            return self.invert_P()
        else:
            # define a function to compute P_inv.dot()
            return Bunch(dot=self.lin_solve)

    def step(self, U_last):
        # return U_last + np.dot(self.P_inv, (self.c - np.dot(self.M,U_last)))
        return U_last + self.P_inv.dot(self.c - self.M.dot(U_last))

    def sweep(self, U_0, k):
        if k == 0:
            return U_0
        U_last = np.copy(U_0)
        for i in range(k):
            U_last[:] = self.step(U_last)[:]
        return U_last

    @property
    def it_matrix(self):
        if self.sparse_format is "array":
            return np.eye(self.P.shape[0])-np.dot(self.P_inv, self.M)
        else:
            func = lambda v: v - self.P_inv.dot(self.M.dot(v))
            it_matrix = Bunch(dot=func)
            return it_matrix

    def lin_solve(self, v):
        return sparse_inv(self.P, v)
