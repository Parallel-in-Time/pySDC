# coding=utf-8
import numpy as np
import scipy.sparse as sprs
from scipy.interpolate import BarycentricInterpolator


def next_neighbors_periodic(p, ps, k):
    """
    Function to find the next neighbors for a periodic setup

    This function gives for a value p the k points next to it which are found in
    in the vector ps and the points which are found periodically.

    Args:
        p: the current point
        ps (np.ndarray): the grid with the potential neighbors
        k (int): number of neighbors to find

    Returns:
        list: the k next neighbors
    """
    p_bar = p - np.floor(p / 1.0) * 1.0
    ps = ps - ps[0]
    distance_to_p = np.asarray(
        list(map(lambda tk: min([np.abs(tk + 1 - p_bar), np.abs(tk - p_bar), np.abs(tk - 1 - p_bar)]), ps)))

    # zip it
    value_index = []
    for d, i in zip(distance_to_p, range(distance_to_p.size)):
        value_index.append((d, i))
    # sort by distance
    value_index_sorted = sorted(value_index, key=lambda s: s[0])
    # take first k indices with least distance and sort them
    return sorted(map(lambda s: s[1], value_index_sorted[0:k]))


def next_neighbors(p, ps, k):
    """
    Function to find the next neighbors for a non-periodic setup

    This function gives for a value p the k points next to it which are found in
    in the vector ps

    Args:
        p: the current point
        ps (np.ndarray): the grid with the potential neighbors
        k (int): number of neighbors to find

    Returns:
        list: the k next neighbors
    """
    distance_to_p = np.abs(ps - p)
    # zip it
    value_index = []
    for d, i in zip(distance_to_p, range(distance_to_p.size)):
        value_index.append((d, i))
    # sort by distance
    value_index_sorted = sorted(value_index, key=lambda s: s[0])
    # take first k indices with least distance and sort them
    return sorted(map(lambda s: s[1], value_index_sorted[0:k]))


def continue_periodic_array(arr, nn):
    """
    Function to append an array for nn neighbors for periodicity

    Args:
        arr (np.ndarray): the input array
        nn (list): the neighbors

    Returns:
        np.ndarray: the continued array
    """

    nn = np.asarray(nn)
    d_nn = nn[1:] - nn[:-1]
    if np.all(d_nn == np.ones(nn.shape[0] - 1)):
        return arr[nn]
    else:
        cont_arr = [arr[nn[0]]]
        shift = 0.
        for n, d in zip(nn[1:], d_nn):
            if d != 1:
                shift = -1
            cont_arr.append(arr[n] + shift)

        return np.asarray(cont_arr)


def restriction_matrix_1d(fine_grid, coarse_grid, k=2, periodic=False, pad=1):
    """
    Function to contruct the restriction matrix in 1d using barycentric interpolation

    Args:
        fine_grid (np.ndarray): a one dimensional 1d array containing the nodes of the fine grid
        coarse_grid (np.ndarray): a one dimensional 1d array containing the nodes of the coarse grid
        k (int): order of the restriction
        periodic (bool): flag to indicate periodicity
        pad (int): padding parameter for boundaries

    Returns:
         sprs.csc_matrix: restriction matrix
    """

    n_g = coarse_grid.size

    if periodic:
        M = np.zeros((coarse_grid.size, fine_grid.size))
        for i, p in zip(range(n_g), coarse_grid):
            nn = next_neighbors_periodic(p, fine_grid, k)
            circulating_one = np.asarray([1.0] + [0.0] * (k - 1))
            cont_arr = continue_periodic_array(fine_grid, nn)
            if p > np.mean(coarse_grid) and not (cont_arr[0] <= p <= cont_arr[-1]):
                cont_arr += 1
            bary_pol = []
            for l in range(k):
                bary_pol.append(BarycentricInterpolator(cont_arr, np.roll(circulating_one, l)))
            M[i, nn] = np.asarray(list(map(lambda x: x(p), bary_pol)))
    else:
        M = np.zeros((coarse_grid.size, fine_grid.size + 2 * pad))
        for i, p in zip(range(n_g), coarse_grid):
            padded_f_grid = border_padding(fine_grid, pad, pad)
            nn = next_neighbors(p, padded_f_grid, k)
            # construct the lagrange polynomials for the k neighbors
            circulating_one = np.asarray([1.0] + [0.0] * (k - 1))
            bary_pol = []
            for l in range(k):
                bary_pol.append(BarycentricInterpolator(padded_f_grid[nn], np.roll(circulating_one, l)))
            M[i, nn] = np.asarray(list(map(lambda x: x(p), bary_pol)))
        if pad > 0:
            M = M[:, pad:-pad]

    return sprs.csc_matrix(M)


def interpolation_matrix_1d(fine_grid, coarse_grid, k=2, periodic=False, pad=1, equidist_nested=True):
    """
    Function to contruct the restriction matrix in 1d using barycentric interpolation

    Args:
        fine_grid (np.ndarray): a one dimensional 1d array containing the nodes of the fine grid
        coarse_grid (np.ndarray): a one dimensional 1d array containing the nodes of the coarse grid
        k (int): order of the restriction
        periodic (bool): flag to indicate periodicity
        pad (int): padding parameter for boundaries
        equidist_nested (bool): shortcut possible, if nodes are equidistant and nested

    Returns:
         sprs.csc_matrix: interpolation matrix
    """

    n_f = fine_grid.size

    if periodic:

        M = np.zeros((fine_grid.size, coarse_grid.size))

        if equidist_nested:

            for i, p in zip(range(n_f), fine_grid):

                if i % 2 == 0:
                    M[i, int(i / 2)] = 1.0
                else:

                    nn = []
                    cpos = int(i / 2)
                    offset = int(k / 2)
                    for j in range(k):
                        nn.append(cpos - offset + 1 + j)
                        if nn[-1] < 0:
                            nn[-1] += coarse_grid.size
                        elif nn[-1] > coarse_grid.size - 1:
                            nn[-1] -= coarse_grid.size
                    nn = sorted(nn)

                    circulating_one = np.asarray([1.0] + [0.0] * (k - 1))
                    if len(nn) > 0:
                        cont_arr = continue_periodic_array(coarse_grid, nn)
                    else:
                        cont_arr = coarse_grid

                    if p > np.mean(fine_grid) and not (cont_arr[0] <= p <= cont_arr[-1]):
                        cont_arr += 1

                    bary_pol = []
                    for l in range(k):
                        bary_pol.append(BarycentricInterpolator(cont_arr, np.roll(circulating_one, l)))
                    M[i, nn] = np.asarray(list(map(lambda x: x(p), bary_pol)))

        else:

            for i, p in zip(range(n_f), fine_grid):
                nn = next_neighbors_periodic(p, coarse_grid, k)
                circulating_one = np.asarray([1.0] + [0.0] * (k - 1))
                cont_arr = continue_periodic_array(coarse_grid, nn)

                if p > np.mean(fine_grid) and not (cont_arr[0] <= p <= cont_arr[-1]):
                    cont_arr += 1

                bary_pol = []
                for l in range(k):
                    bary_pol.append(BarycentricInterpolator(cont_arr, np.roll(circulating_one, l)))
                M[i, nn] = np.asarray(list(map(lambda x: x(p), bary_pol)))

    else:

        M = np.zeros((fine_grid.size, coarse_grid.size + 2 * pad))
        padded_c_grid = border_padding(coarse_grid, pad, pad)

        if equidist_nested:

            for i, p in zip(range(n_f), fine_grid):

                if i % 2 != 0:
                    M[i, int((i - 1) / 2) + 1] = 1.0
                else:
                    nn = []
                    cpos = int(i / 2)
                    offset = int(k / 2)
                    for j in range(k):
                        nn.append(cpos - offset + 1 + j)
                        if nn[-1] < 0:
                            nn[-1] += k
                        elif nn[-1] > coarse_grid.size + 1:
                            nn[-1] -= k
                    nn = sorted(nn)
                    # construct the lagrange polynomials for the k neighbors
                    circulating_one = np.asarray([1.0] + [0.0] * (k - 1))
                    bary_pol = []
                    for l in range(k):
                        bary_pol.append(BarycentricInterpolator(padded_c_grid[nn], np.roll(circulating_one, l)))
                    M[i, nn] = np.asarray(list(map(lambda x: x(p), bary_pol)))

        else:

            for i, p in zip(range(n_f), fine_grid):
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
    """
    Function to pad/embed an array at the boundaries

    Args:
        grid (np.npdarray): the input array
        l: left boundary
        r: right boundary
        pad_type: type of padding

    Returns:
        np.npdarray: the padded array

    """

    assert l < grid.size and r < grid.size
    padded_arr = np.zeros(grid.size + l + r)
    if pad_type == 'mirror':
        for i in range(l):
            padded_arr[i] = 2 * grid[0] - grid[l - i]
        for j in range(r):
            padded_arr[-j - 1] = 2 * grid[-1] - grid[-r + j - 1]
    padded_arr[l:l + grid.size] = grid
    return padded_arr
