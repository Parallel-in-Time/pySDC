import pytest
import numpy as np
from numpy.polynomial.polynomial import polyval

from pySDC.core.Collocation import CollBase
import pySDC.helpers.transfer_helper as th

t_start = np.random.rand(1) * 0.2
t_end = 0.8 + np.random.rand(1) * 0.2

node_types = ['EQUID', 'LEGENDRE']
quad_types = ['GAUSS', 'LOBATTO', 'RADAU-RIGHT', 'RADAU-LEFT']


@pytest.mark.base
@pytest.mark.parametrize("node_type", node_types)
@pytest.mark.parametrize("quad_type", quad_types)
def test_Q_transfer(node_type, quad_type):
    """
    A simple test program to check the order of the Q interpolation/restriction
    """

    for M in range(3, 9):
        Mfine = M
        Mcoarse = int((Mfine + 1) / 2.0)

        coll_fine = CollBase(Mfine, 0, 1, node_type=node_type, quad_type=quad_type)
        coll_coarse = CollBase(Mcoarse, 0, 1, node_type=node_type, quad_type=quad_type)

        assert (
            coll_fine.left_is_node == coll_coarse.left_is_node
        ), 'ERROR: should be using the same class for coarse and fine Q'

        fine_grid = coll_fine.nodes
        coarse_grid = coll_coarse.nodes

        for order in range(2, coll_coarse.num_nodes + 1):
            Pcoll = th.interpolation_matrix_1d(fine_grid, coarse_grid, k=order, pad=0, equidist_nested=False)
            Rcoll = th.restriction_matrix_1d(fine_grid, coarse_grid, k=order, pad=0)

            for polyorder in range(1, order + 2):
                coeff = np.random.rand(polyorder)
                ufine = polyval(fine_grid, coeff)
                ucoarse = polyval(coarse_grid, coeff)

                uinter = Pcoll.dot(ucoarse)
                urestr = Rcoll.dot(ufine)

                err_inter = np.linalg.norm(uinter - ufine, np.inf)
                err_restr = np.linalg.norm(urestr - ucoarse, np.inf)

                if polyorder <= order:
                    assert err_inter < 5e-15, "ERROR: Q-interpolation order is not reached, got %s" % err_inter
                    assert err_restr < 3e-15, "ERROR: Q-restriction order is not reached, got %s" % err_restr
                else:
                    assert err_inter > 2e-15, "ERROR: Q-interpolation order is higher than expected, got %s" % polyorder


@pytest.mark.base
@pytest.mark.parametrize("node_type", node_types)
@pytest.mark.parametrize("quad_type", quad_types)
def test_Q_transfer_minimal(node_type, quad_type):
    """
    A simple test program to check the order of the Q interpolation/restriction for only 2 coarse nodes
    """

    Mcoarse = 2
    coll_coarse = CollBase(Mcoarse, 0, 1, node_type=node_type, quad_type=quad_type)

    for M in range(3, 9):
        Mfine = M

        coll_fine = CollBase(Mfine, 0, 1, node_type=node_type, quad_type=quad_type)

        assert (
            coll_fine.left_is_node == coll_coarse.left_is_node
        ), 'ERROR: should be using the same class for coarse and fine Q'

        fine_grid = coll_fine.nodes
        coarse_grid = coll_coarse.nodes

        Pcoll = th.interpolation_matrix_1d(fine_grid, coarse_grid, k=2, pad=0, equidist_nested=False)
        Rcoll = th.restriction_matrix_1d(fine_grid, coarse_grid, k=2, pad=0)

        for polyorder in range(1, 3):
            coeff = np.random.rand(polyorder)
            ufine = polyval(fine_grid, coeff)
            ucoarse = polyval(coarse_grid, coeff)

            uinter = Pcoll.dot(ucoarse)
            urestr = Rcoll.dot(ufine)

            err_inter = np.linalg.norm(uinter - ufine, np.inf)
            err_restr = np.linalg.norm(urestr - ucoarse, np.inf)

            if polyorder <= 2:
                assert err_inter < 2e-15, "ERROR: Q-interpolation order is not reached, got %s" % err_inter
                assert err_restr < 2e-15, "ERROR: Q-restriction order is not reached, got %s" % err_restr
            else:
                assert err_inter > 2e-15, "ERROR: Q-interpolation order is higher than expected, got %s" % polyorder
