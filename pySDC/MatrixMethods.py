# coding=utf-8
import numpy as np
import scipy as sp
import scipy.sparse as sprs
import scipy.sparse.linalg as spla
import scipy.linalg as la
import scipy.interpolate as intpl
from scipy.integrate import quad
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import matplotlib as mplt
import pylab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pySDC.tools.transfer_tools import to_sparse
import functools as ft
# Own collection of tools, require the imports from above. But first we will try to
# to use and enhance the tools given by pySDC
#from pint_matrix_tools import *


# We start with a bunch of helperfunctions

def prepare_function(f):
    """
    This prepares a function for a simple use of numpy arrays to get a vector f(numpy.array) = numpy.array

    :param f: A function
    :return: Prepared function
    """
    return np.vectorize(f)


def extract_vector(f, u):
    """
    We expect a function which has arbitrary many arguments f(t,x,y,z, ...),
    the convention is that the first is argument is always time

    :param f: a function f(t,x,y,z,...)
    :param u: u is a list of numpy.arrays for each dimension containing the values one considers to evaluate
    :return: vector with the evaluated rhs
    """
    if len(u) is 1:
        return prepare_function(f)(u[0])
    else:
        vect_list = []
        next_u = u[1:]
        for x in u[0]:
            next_f = ft.partial(f, x)
            vect_list.append(extract_vector(next_f, next_u))
        return np.concatenate(vect_list)


def distributeToFirst(v, N):
    """
    Distribute to first, fill up with zeros
    :param v: numpy vector
    :param N: number of times shape is used
    :return: V=(v,0,...,0)
    """
    z = np.zeros(v.shape)
    vlist = [v]+[z]*(N-1)
    return np.concatenate(vlist)


def distributeToAll(v, N):
    """
    Distribute to all
    :param v: numpy vector
    :param N: number of times shape is repeated
    :return: V=(v,v,...,v)
    """
    vlist = [v]*(N)
    return np.concatenate(vlist)


def transform_to_unit_interval(x,t_l,t_r):
    return (x-t_l)/(t_r-t_l)

def sparse_inv(P,v):
    """
    Uses sparse solver to compute P^-1 * v
    :param P: sparse_matrix
    :param v: dense vector
    :return:
    """
    return spla.spsolve(P,v)
# now some classes building differrent classes of Linear Iterative Solver, using sparse matrices

class DenseIterativeSolver(object):
    """ The basic Iterative Solver class,
        several steps of the iterative solver are called sweeps"""
    def __init__(self,P,M,c,format="dense"):
        assert P.shape == M.shape and c.shape[0] == M.shape[0], "Matrix P and matrix M don't fit"
        self.format = format
        self.P = to_sparse(P, format)
        self.M = to_sparse(M, format)
        self.c = to_sparse(c, format)
        if format is "dense":
            self.P_inv = self.invert_P()
        else:
            # define a function to compute P_inv.dot()
            self.P_inv.dot = ft.partial(sparse_inv, P=self.P)

    def invert_P(self):
        return la.inv(self.P)

    def step(self,U_last):
        # return U_last + np.dot(self.P_inv, (self.c - np.dot(self.M,U_last)))
        return U_last + self.P_inv.dot(self.c - self.M.dot(U_last))

    def sweep(self,U_0,k):
        if k==0:
            return U_0
        U_last = np.copy(U_0)
        for i in range(k):
            U_last[:] = self.step(U_last)[:]
        return U_last

    def it_matrix(self):
        if self.format is "dense":
            return np.eye(self.P.shape[0])-np.dot(self.P_inv, self.M)
        else:
            func = lambda v: v-self.P_inv.dot(self.M.dot(v))
            it_matrix = None
            it_matrix.dot = func
            return it_matrix


class SparseIterativeSolver(object):
    """
    The basic iterative solver class , using sparse matrices.
    """
