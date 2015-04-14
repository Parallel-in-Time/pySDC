# coding=utf-8
import numpy as np
import scipy as sp
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

