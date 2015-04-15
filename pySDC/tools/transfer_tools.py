# coding=utf-8
import numpy as np
import scipy.interpolate as intpl
import scipy.sparse as sprs


def to_sparse(D, format="csc"):
    """
    Transform dense matrix to sparse matrix of return_type
        bsr_matrix(arg1[, shape, dtype, copy, blocksize]) 	Block Sparse Row matrix
        coo_matrix(arg1[, shape, dtype, copy]) 	A sparse matrix in COOrdinate format.
        csc_matrix(arg1[, shape, dtype, copy]) 	Compressed Sparse Column matrix
        csr_matrix(arg1[, shape, dtype, copy]) 	Compressed Sparse Row matrix
        dia_matrix(arg1[, shape, dtype, copy]) 	Sparse matrix with DIAgonal storage
        dok_matrix(arg1[, shape, dtype, copy]) 	Dictionary Of Keys based sparse matrix.
        lil_matrix(arg1[, shape, dtype, copy]) 	Row-based linked list sparse matrix
    :param D: Dense matrix
    :param format: how to save the sparse matrix
    :return: sparse version
    """
    if format == "bsr":
        return sprs.bsr_matrix(D)
    elif format == "coo":
        return sprs.coo_matrix(D)
    elif format == "csc":
        return sprs.csc_matrix(D)
    elif format == "csr":
        return sprs.csr_matrix(D)
    elif format == "dia":
        return sprs.dia_matrix(D)
    elif format == "dok":
        return sprs.dok_matrix(D)
    elif format == "lil":
        return sprs.lil_matrix(D)
    else:
        return D


def next_neighbors(p, ps, k):
    """
    This function gives for a value p the k points next to it which are found in
    in the vector ps
    :param p: value
    :param ps: ndarray, vector where to find the next neighbors
    :param k: integer, number of neighbours
    :return: ndarray, with the k next neighbors
    """
    distance_to_p = np.abs(ps-p)
    # zip it
    value_index = []
    for d,i in zip(distance_to_p, range(distance_to_p.size)):
        value_index.append((d,i))
    # sort by distance
    value_index_sorted = sorted(value_index, key=lambda s: s[0])
    # take first k indices with least distance and sort them
    return sorted(map(lambda s: s[1], value_index_sorted[0:k]))


def restriction_matrix_1d(fine_grid, coarse_grid, k=2, return_type="csc"):
    """
    We construct the restriction matrix between two 1d grids, using lagrange interpolation.

    :param fine_grid: a one dimensional 1d array containing the nodes of the fine grid
    :param coarse_grid: a one dimensional 1d array containing the nodes of the coarse grid
    :param k: order of the restriction
    :return: a restriction matrix
    """
    M = np.zeros((coarse_grid.size, fine_grid.size))
    n_g = coarse_grid.size

    for i, p in zip(range(n_g), coarse_grid):
        nn = next_neighbors(p, fine_grid, k)
        # construct the lagrange polynomials for the k neighbors
        circulating_one = np.asarray([1.0]+[0.0]*(k-1))
        lag_pol = []
        for l in range(k):
            lag_pol.append(intpl.lagrange(fine_grid[nn], np.roll(circulating_one, l)))
        M[i, nn] = np.asarray(map(lambda x: x(p), lag_pol))

    return to_sparse(M, return_type)


def interpolation_matrix_1d(fine_grid, coarse_grid, k=2, return_type="csc"):
    """
    We construct the interpolation matrix between two 1d grids, using lagrange interpolation.

    :param fine_grid: a one dimensional 1d array containing the nodes of the fine grid
    :param coarse_grid: a one dimensional 1d array containing the nodes of the coarse grid
    :param k: order of the restriction
    :return: a interpolation matrix
    """
    M = np.zeros((coarse_grid.size, fine_grid.size))
    n_f = fine_grid.size

    for i, p in zip(range(n_f), fine_grid):
        nn = next_neighbors(p, coarse_grid, k)
        # construct the lagrange polynomials for the k neighbors
        circulating_one = np.asarray([1.0]+[0.0]*(k-1))
        lag_pol = []
        for l in range(k):
            lag_pol.append(intpl.lagrange(coarse_grid[nn], np.roll(circulating_one, l)))
        M[i, nn] = np.asarray(map(lambda x: x(p), lag_pol))
    return to_sparse(M, return_type)


def kron_on_list(matrix_list):
    """
    :param matrix_list: a list of sparse matrices
    :return: a matrix
    """
    if len(matrix_list) == 2:
        return sprs.kron(matrix_list[0], matrix_list[1])
    elif len(matrix_list) == 1:
        return matrix_list[0]
    else:
        return sprs.kron(matrix_list[0], kron_on_list(matrix_list[1:]))


