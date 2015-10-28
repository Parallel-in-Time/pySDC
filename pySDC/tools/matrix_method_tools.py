# coding=utf-8
import numpy as np
import scipy.sparse.linalg as spla
import functools as ft



class Bunch:
    """
    Create an object(Bunch) with some Attributes you initialize in the beginning.

    """
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


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
