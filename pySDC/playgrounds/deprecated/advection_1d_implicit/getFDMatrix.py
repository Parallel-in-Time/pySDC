import numpy as np
import scipy.linalg as LA
import scipy.sparse as sp


def getFDMatrix(N, order, dx):
    if order == 1:
        stencil = [-1.0, 1.0]
        zero_pos = 2
        coeff = 1.0

    elif order == 2:
        stencil = [1.0, -4.0, 3.0]
        coeff = 1.0 / 2.0
        zero_pos = 3

    elif order == 3:
        stencil = [1.0, -6.0, 3.0, 2.0]
        coeff = 1.0 / 6.0
        zero_pos = 3

    elif order == 4:
        stencil = [-5.0, 30.0, -90.0, 50.0, 15.0]
        coeff = 1.0 / 60.0
        zero_pos = 4

    elif order == 5:
        stencil = [3.0, -20.0, 60.0, -120.0, 65.0, 12.0]
        coeff = 1.0 / 60.0
        zero_pos = 5
    else:
        print("Order " + order + " not implemented.")

    first_col = np.zeros(N)

    # Because we need to specific first column (not row) in circulant, flip stencil array
    first_col[0 : np.size(stencil)] = np.flipud(stencil)

    # Circulant shift of coefficient column so that entry number zero_pos becomes first entry
    first_col = np.roll(first_col, -np.size(stencil) + zero_pos, axis=0)

    return sp.csc_matrix(coeff * (1.0 / dx) * LA.circulant(first_col))
