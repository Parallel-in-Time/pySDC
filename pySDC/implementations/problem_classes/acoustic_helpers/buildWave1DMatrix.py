import numpy as np
import scipy.sparse as sp

from pySDC.helpers.problem_helper import get_finite_difference_matrix

wave_order = 6


def getFDMatrix(N, dx, order, stencil_type, bc):
    """
    First derivative matrix. Only homogeneous boundary conditions are supported, so the
    right hand side contribution returned by the helper has to vanish.
    """
    A, b = get_finite_difference_matrix(
        derivative=1, order=order, stencil_type=stencil_type, dx=dx, size=N, dim=1, bc=bc
    )
    assert np.linalg.norm(b) == 0.0, 'Inhomogeneous boundary conditions are not supported here'
    return sp.csc_matrix(A)


def getWave1DMatrix(N, dx, bc_left, bc_right):
    Id = sp.eye(2 * N)

    D_u = getFDMatrix(N, dx, wave_order, 'center', (bc_left[0], bc_right[0]))
    D_p = getFDMatrix(N, dx, wave_order, 'center', (bc_left[1], bc_right[1]))
    Zero = np.zeros((N, N))
    M1 = sp.hstack((Zero, D_p), format="csc")
    M2 = sp.hstack((D_u, Zero), format="csc")
    M = sp.vstack((M1, M2), format="csc")
    return sp.csc_matrix(Id), sp.csc_matrix(M)


def getWave1DAdvectionMatrix(N, dx, order):
    Dx = getFDMatrix(N, dx, order, 'upwind', 'periodic')
    Zero = np.zeros((N, N))
    M1 = sp.hstack((Dx, Zero), format="csc")
    M2 = sp.hstack((Zero, Dx), format="csc")
    M = sp.vstack((M1, M2), format="csc")
    return sp.csc_matrix(M)
