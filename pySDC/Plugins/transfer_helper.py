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
        return to_dense(D)


def to_dense(D):
    if sprs.issparse(D):
        return D.toarray()
    elif isinstance(D, np.ndarray):
        return D

def next_neighbors_periodic(p, ps, k, T=None):
    """
    This function gives for a value p the k points next to it which are found in
    in the vector ps and the points which are found periodically.
    :param p: value
    :param ps: ndarray, vector where to find the next neighbors
    :param k: integer, number of neighbours
    :return: ndarray, with the k next neighbors
    """
    if T is None:
        T = ps[-1]-2*ps[0]+ps[1]
    p_bar = p - np.floor(p/T)*T
    ps = ps - ps[0]
    distance_to_p = np.asarray(list(map(lambda tk: min([np.abs(tk+T-p_bar), np.abs(tk-p_bar), np.abs(tk-T-p_bar)]),ps)))
    # print p_bar
    # print distance_to_p
    # zip it
    value_index = []
    for d,i in zip(distance_to_p, range(distance_to_p.size)):
        value_index.append((d, i))
    # sort by distance
    value_index_sorted = sorted(value_index, key=lambda s: s[0])
    # take first k indices with least distance and sort them
    return sorted(list(map(lambda s: s[1], value_index_sorted[0:k])))


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
    neighbors = sorted(list(map(lambda s: s[1], value_index_sorted[0:k])))
    if neighbors[0] == 0:
        neighbors = neighbors[1:]
    if neighbors[-1] == 1:
        neighbors = neighbors[:-1]
    return neighbors



def continue_periodic_array(arr,nn,T):
    nn = np.asarray(nn)
    d_nn = nn[1:]-nn[:-1]
    if np.all(d_nn == np.ones(nn.shape[0]-1)):
        return arr[nn]
    else:
        cont_arr = [arr[nn[0]]]
        shift = 0.
        for n,d in zip(nn[1:],d_nn):
            if d != 1:
                shift = -T
            cont_arr.append(arr[n]+shift)

        return np.asarray(cont_arr)


def restriction_matrix_1d(fine_grid, coarse_grid, k=2, return_type="csc", periodic=False, T=1.0):
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
        if periodic:
            nn = next_neighbors_periodic(p, fine_grid, k, T)
            circulating_one = np.asarray([1.0]+[0.0]*(k-1))
            lag_pol = []
            cont_arr = continue_periodic_array(fine_grid, nn, T)
            if p > np.mean(coarse_grid) and not (p >= cont_arr[0] and p <= cont_arr[-1]):
                cont_arr = cont_arr + T
            for l in range(k):
                lag_pol.append(intpl.lagrange(cont_arr, np.roll(circulating_one, l)))
            M[i, nn] = np.asarray(list(map(lambda x: x(p), lag_pol)))
        else:
            nn = next_neighbors(p, fine_grid, k)
        # construct the lagrange polynomials for the k neighbors
            circulating_one = np.asarray([1.0]+[0.0]*(k-1))
            lag_pol = []
            for l in range(k):
                lag_pol.append(intpl.lagrange(fine_grid[nn], np.roll(circulating_one, l)))
            M[i, nn] = np.asarray(list(map(lambda x: x(p), lag_pol)))

    return to_sparse(M, return_type)


def interpolation_matrix_1d(fine_grid, coarse_grid, k=2, return_type="csc", periodic=False, T=1.0):
    """
    We construct the interpolation matrix between two 1d grids, using lagrange interpolation.
    :param fine_grid: a one dimensional 1d array containing the nodes of the fine grid
    :param coarse_grid: a one dimensional 1d array containing the nodes of the coarse grid
    :param k: order of the restriction
    :return: a interpolation matrix
    """
    M = np.zeros((fine_grid.size, coarse_grid.size))
    n_f = fine_grid.size

    for i, p in zip(range(n_f), fine_grid):
        if periodic:
            nn = next_neighbors_periodic(p, coarse_grid, k, T)
            circulating_one = np.asarray([1.0]+[0.0]*(k-1))
            lag_pol = []
            cont_arr = continue_periodic_array(coarse_grid, nn, T)

            if p > np.mean(fine_grid) and not (p >= cont_arr[0] and p <= cont_arr[-1]):
                cont_arr = cont_arr + T
            # print cont_arr

            for l in range(k):
                lag_pol.append(intpl.lagrange(cont_arr, np.roll(circulating_one, l)))
            M[i, nn] = np.asarray(list(map(lambda x: x(p), lag_pol)))
        else:
            nn = next_neighbors(p, coarse_grid, k)
            print(i,nn)
        # construct the lagrange polynomials for the k neighbors
            circulating_one = np.asarray([1.0]+[0.0]*(k-1))
            lag_pol = []
            for l in range(k):
                lag_pol.append(intpl.lagrange(coarse_grid[nn], np.roll(circulating_one, l)))
            M[i, nn] = np.asarray(list(map(lambda x: x(p), lag_pol)))
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


def matrixN(tau, rows=-1, last_value=1.0):
    n = tau.shape[0]
    if rows == -1:
        rows = n
    N = np.zeros((rows, n))
    # construct the lagrange polynomials
    circulating_one = np.asarray([1.0]+[0.0]*(n-1))
    lag_pol = []
    for i in range(n):
        lag_pol.append(intpl.lagrange(tau, np.roll(circulating_one, i)))
        N[:, i] = -np.ones(rows)*lag_pol[-1](last_value)
    return N

def interpolate_to_t_end(nodes_on_unit, values):
    """
    Assume a GaussLegendre nodes, we are interested in the value at the end of
    the interval, but we now only the values in the interior of the interval.
    We compute the value by legendre interpolation.
    :param nodes_on_unit: nodes transformed to the unit interval
    :param values: values on those nodes
    :return: interpolation to the end of the interval
    """
    n = nodes_on_unit.shape[0]
    circulating_one = np.asarray([1.0]+[0.0]*(n-1))
    lag_pol = []
    result = np.zeros(values[0].shape)
    for i in range(n):
        lag_pol.append(intpl.lagrange(nodes_on_unit, np.roll(circulating_one, i)))
        result += values[i]*lag_pol[-1](1.0)
    return result