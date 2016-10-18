# coding=utf-8
import numpy as np
import scipy.sparse as sprs
from scipy.interpolate import BarycentricInterpolator


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
    return sorted(map(lambda s: s[1], value_index_sorted[0:k]))


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


def restriction_matrix_1d(fine_grid, coarse_grid, k=2, periodic=False, pad=1, T=1.0):
    """
    We construct the restriction matrix between two 1d grids, using lagrange interpolation.
    :param fine_grid: a one dimensional 1d array containing the nodes of the fine grid
    :param coarse_grid: a one dimensional 1d array containing the nodes of the coarse grid
    :param k: order of the restriction
    :return: a restriction matrix
    """

    n_g = coarse_grid.size

    if periodic:
        M = np.zeros((coarse_grid.size, fine_grid.size))
        for i, p in zip(range(n_g), coarse_grid):
            nn = next_neighbors_periodic(p, fine_grid, k, T)
            circulating_one = np.asarray([1.0]+[0.0]*(k-1))
            cont_arr = continue_periodic_array(fine_grid, nn, T)
            if p > np.mean(coarse_grid) and not (p >= cont_arr[0] and p <= cont_arr[-1]):
                cont_arr = cont_arr + T
            bary_pol = []
            for l in range(k):
                bary_pol.append(BarycentricInterpolator(cont_arr, np.roll(circulating_one, l)))
            M[i, nn] = np.asarray(list(map(lambda x: x(p), bary_pol)))
    else:
        M = np.zeros((coarse_grid.size, fine_grid.size+2*pad))
        for i, p in zip(range(n_g), coarse_grid):
            padded_f_grid = border_padding(fine_grid, pad, pad)
            nn = next_neighbors(p, padded_f_grid, k)
            # construct the lagrange polynomials for the k neighbors
            circulating_one = np.asarray([1.0]+[0.0]*(k-1))
            bary_pol = []
            for l in range(k):
                bary_pol.append(BarycentricInterpolator(padded_f_grid[nn], np.roll(circulating_one, l)))
            M[i, nn] = np.asarray(list(map(lambda x: x(p), bary_pol)))
        if pad > 0:
            M = M[:, pad:-pad]

    return sprs.csc_matrix(M)

#
def interpolation_matrix_1d(fine_grid, coarse_grid, k=2, periodic=False, pad=1, T=1.0):
    """
    We construct the interpolation matrix between two 1d grids, using lagrange interpolation.
    :param fine_grid: a one dimensional 1d array containing the nodes of the fine grid
    :param coarse_grid: a one dimensional 1d array containing the nodes of the coarse grid
    :param k: order of the interpolation
    :return: a interpolation matrix
    """

    n_f = fine_grid.size

    if periodic:
        M = np.zeros((fine_grid.size, coarse_grid.size))
        for i, p in zip(range(n_f), fine_grid):
            nn = next_neighbors_periodic(p, coarse_grid, k, T)
            circulating_one = np.asarray([1.0]+[0.0]*(k-1))
            cont_arr = continue_periodic_array(coarse_grid, nn, T)

            if p > np.mean(fine_grid) and not (p >= cont_arr[0] and p <= cont_arr[-1]):
                cont_arr = cont_arr + T

            bary_pol = []
            for l in range(k):
                bary_pol.append(BarycentricInterpolator(cont_arr, np.roll(circulating_one, l)))
            M[i, nn] = np.asarray(list(map(lambda x: x(p), bary_pol)))
    else:
        M = np.zeros((fine_grid.size, coarse_grid.size+2*pad))
        for i, p in zip(range(n_f), fine_grid):
            padded_c_grid = border_padding(coarse_grid, pad, pad)
            nn = next_neighbors(p, padded_c_grid, k)
            # construct the lagrange polynomials for the k neighbors
            circulating_one = np.asarray([1.0] + [0.0] * (k - 1))
            bary_pol = []
            for l in range(k):
                bary_pol.append(BarycentricInterpolator(padded_c_grid[nn], np.roll(circulating_one, l)))
            M[i, nn] = np.asarray(list(map(lambda x: x(p), bary_pol)))
        if pad > 0:
            M = M[:, pad:-pad]
    return sprs.csc_matrix(M)


def border_padding(grid, l, r, pad_type='mirror'):
    """ returns an array where the original array is embedded and the borders are enhanced by
        a certain padding strategy, e.g. mirroring the distances
    """
    assert l < grid.size and r < grid.size
    padded_arr = np.zeros(grid.size + l + r)
    if pad_type is 'mirror':
        for i in range(l):
            padded_arr[i] = 2 * grid[0] - grid[l - i]
        for j in range(r):
            padded_arr[-j - 1] = 2 * grid[-1] - grid[-r + j - 1]
    padded_arr[l:l+grid.size] = grid
    return padded_arr