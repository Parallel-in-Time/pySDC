import numpy as np
import nose
from numpy.polynomial.polynomial import polyval

import pySDC.helpers.transfer_helper as th

from tests.test_helpers import get_derived_from_in_package
from pySDC.core.Collocation import CollBase

classes = []

def setup():
    global classes, t_start, t_end

    # generate random boundaries for the time slice with 0.0 <= t_start < 0.2 and 0.8 <= t_end < 1.0
    t_start = np.random.rand(1) * 0.2
    t_end = 0.8 + np.random.rand(1) * 0.2
    classes = get_derived_from_in_package(CollBase, 'pySDC/implementations/collocation_classes')

@nose.tools.with_setup(setup)
def test_Q_transfer():
    for collclass in classes:
        yield check_Q_transfer, collclass

def check_Q_transfer(collclass):
    """
    A simple test program to check the order of the Q interpolation/restriction
    """

    for M in range(3, 9):

        Mfine = M
        Mcoarse = int((Mfine+1)/2.0)

        coll_fine = collclass(Mfine, 0, 1)
        coll_coarse = collclass(Mcoarse, 0, 1)

        assert coll_fine.left_is_node == coll_coarse.left_is_node, 'ERROR: should be using the same class for coarse and fine Q'

        if not coll_fine.left_is_node:
            fine_grid = np.concatenate(([0], coll_fine.nodes))
            coarse_grid = np.concatenate(([0], coll_coarse.nodes))
        else:
            fine_grid = coll_fine.nodes
            coarse_grid = coll_coarse.nodes

        for order in range(2,coll_coarse.num_nodes+1):

            Pcoll = th.interpolation_matrix_1d(fine_grid, coarse_grid, k=order, pad=0)
            Rcoll = th.restriction_matrix_1d(fine_grid, coarse_grid, k=order, pad=0)

            for polyorder in range(1,order+2):
                coeff = np.random.rand(polyorder)
                ufine = polyval(fine_grid,coeff)
                ucoarse = polyval(coarse_grid,coeff)

                uinter = Pcoll.dot(ucoarse)
                urestr = Rcoll.dot(ufine)

                err_inter = np.linalg.norm(uinter-ufine, np.inf)
                err_restr = np.linalg.norm(urestr-ucoarse, np.inf)

                if polyorder <= order:
                    assert err_inter < 2E-15, "ERROR: Q-interpolation order is not reached, got %s" %err_inter
                    assert err_restr < 2E-15, "ERROR: Q-restriction order is not reached, got %s" % err_restr
                else:
                    assert err_inter > 2E-15, "ERROR: Q-interpolation order is higher than expected, got %s" % polyorder
                    # if Mfine != 5 and Mfine != 3:
                    #     assert err_restr > 2E-15, "ERROR: Q-restriction order is higher than expected, got %s" % polyorder

