import numpy as np
import pytest

import pySDC.helpers.transfer_helper as th


def periodic_grid(nvars):
    """Equidistant periodic grid, on the unit period that the periodic helpers assume."""
    return np.arange(nvars) / nvars


def measure_order(errors):
    """Order of accuracy from errors obtained at successively doubled resolutions."""
    errors = np.array(errors)
    return np.log(errors[:-1] / errors[1:]) / np.log(2)


@pytest.mark.base
def test_next_neighbors_periodic():
    """
    Near either end of the grid the neighbourhood has to wrap around to the other end. The indices
    come back sorted, so a wrapped neighbourhood shows up as points from both ends.
    """
    grid = periodic_grid(16)

    assert th.next_neighbors_periodic(grid[0], grid, 2) == [0, 1]
    assert th.next_neighbors_periodic(grid[0], grid, 4) == [0, 1, 2, 15]
    assert th.next_neighbors_periodic(grid[0], grid, 6) == [0, 1, 2, 3, 14, 15]

    # a point in the last cell, between the last grid point and the wrapped-around first one
    assert th.next_neighbors_periodic(15.5 / 16, grid, 2) == [0, 15]
    assert th.next_neighbors_periodic(15.5 / 16, grid, 4) == [0, 1, 14, 15]

    # in the middle of the grid the periodicity makes no difference
    for k in [2, 4, 6]:
        assert th.next_neighbors_periodic(grid[8], grid, k) == th.next_neighbors(grid[8], grid, k)


@pytest.mark.base
@pytest.mark.parametrize('k', [2, 4])
@pytest.mark.parametrize('ratio', [2, 3])
def test_interpolation_matrix_1d_periodic(k, ratio):
    """
    The general periodic interpolation, which is what you get without the `equidist_nested`
    shortcut. It has to reach order `k` for any refinement ratio, not just for nested grids.
    """
    errors = []
    for nvars_coarse in [8, 16, 32, 64]:
        coarse, fine = periodic_grid(nvars_coarse), periodic_grid(nvars_coarse * ratio)
        P = th.interpolation_matrix_1d(fine, coarse, k=k, periodic=True, equidist_nested=False)

        assert np.allclose(P.toarray() @ np.ones(coarse.size), 1.0), 'Does not reproduce a constant'
        errors.append(np.max(np.abs(P.dot(np.sin(2 * np.pi * coarse)) - np.sin(2 * np.pi * fine))))

    order = measure_order(errors)
    assert np.isclose(np.median(order), k, atol=0.3), f'Expected order {k}, got {np.median(order):.2f}'


@pytest.mark.base
@pytest.mark.parametrize('k', [2, 4])
def test_interpolation_matrix_1d_nested_shortcut(k):
    """
    `equidist_nested` only skips work: on grids that really are equidistant and nested it has to
    produce the same matrix as the general path.
    """
    coarse, fine = periodic_grid(16), periodic_grid(32)
    shortcut = th.interpolation_matrix_1d(fine, coarse, k=k, periodic=True, equidist_nested=True)
    general = th.interpolation_matrix_1d(fine, coarse, k=k, periodic=True, equidist_nested=False)

    assert np.allclose(shortcut.toarray(), general.toarray()), 'The shortcut is not the general path'


@pytest.mark.base
@pytest.mark.parametrize('k', [2, 4])
def test_restriction_matrix_1d_periodic(k):
    """
    Periodic restriction. It is interpolation the other way around, so on nested grids, where every
    coarse point sits on a fine one, it has to collapse to injection.
    """
    coarse, fine = periodic_grid(16), periodic_grid(32)
    R = th.restriction_matrix_1d(fine, coarse, k=k, periodic=True)

    assert R.nnz == coarse.size, 'On nested grids every coarse point should take a single fine value'
    assert np.allclose(R.toarray(), np.eye(fine.size)[::2]), 'Does not collapse to injection'

    # the same construction as interpolating from the fine grid onto the coarse one
    swapped = th.interpolation_matrix_1d(coarse, fine, k=k, periodic=True, equidist_nested=False)
    assert np.allclose(R.toarray(), swapped.toarray()), 'Does not match interpolation with the grids swapped'

    # on grids that do not share their points, restriction has to reach order k like interpolation
    errors = []
    for m in [4, 8, 16, 32]:
        coarse, fine = periodic_grid(5 * m), periodic_grid(12 * m)
        R = th.restriction_matrix_1d(fine, coarse, k=k, periodic=True)

        assert np.allclose(R.toarray() @ np.ones(fine.size), 1.0), 'Does not reproduce a constant'
        errors.append(np.max(np.abs(R.dot(np.sin(2 * np.pi * fine)) - np.sin(2 * np.pi * coarse))))

    order = measure_order(errors)
    assert np.isclose(np.median(order), k, atol=0.3), f'Expected order {k}, got {np.median(order):.2f}'


@pytest.mark.base
def test_restriction_matrix_1d_padding():
    """
    Away from a periodic boundary the fine grid is padded, and the columns belonging to the ghost
    points are dropped again so that the values outside the domain count as zero. That only bites
    when the stencil is wide enough to reach past the boundary.
    """
    nvars_fine, nvars_coarse, k = 16, 7, 8
    fine = (np.arange(nvars_fine) + 1) / (nvars_fine + 1)
    coarse = (np.arange(nvars_coarse) + 1) / (nvars_coarse + 1)

    unpadded = th.restriction_matrix_1d(fine, coarse, k=k, pad=0).toarray()
    padded = th.restriction_matrix_1d(fine, coarse, k=k, pad=1).toarray()

    assert padded.shape == (nvars_coarse, nvars_fine), 'Padding must not leave the ghost columns behind'

    differs = np.abs(padded - unpadded).max(axis=1) > 1e-12
    assert list(differs) == [True] + [False] * (nvars_coarse - 2) + [True], 'Only the boundary rows may change'

    row_sums = padded @ np.ones(nvars_fine)
    assert np.allclose(row_sums[1:-1], 1.0), 'The interior still has to reproduce a constant'
    assert np.all(row_sums[[0, -1]] < 1.0), 'The boundary rows drop the weight of the ghost points'


if __name__ == '__main__':
    test_next_neighbors_periodic()
