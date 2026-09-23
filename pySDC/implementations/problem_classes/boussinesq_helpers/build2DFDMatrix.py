import numpy as np
import scipy.sparse as sp

from pySDC.implementations.problem_classes.boussinesq_helpers.buildFDMatrix import (
    getMatrix,
    getUpwindMatrix,
)


#
#
#
def get2DUpwindMatrix(N, dx, order):
    Dx = getUpwindMatrix(N[0], dx, order)
    return sp.kron(Dx, sp.eye(N[1]), format="csr")


#
#
#
def get2DMesh(N, x_b, z_b, bc_hor, bc_ver):
    assert np.size(N) == 2, 'N needs to be an array with two entries: N[0]=Nx and N[1]=Nz'
    assert (
        np.size(x_b) == 2
    ), 'x_b needs to be an array with two entries: x_b[0] = left boundary, x_b[1] = right boundary'
    assert (
        np.size(z_b) == 2
    ), 'z_b needs to be an array with two entries: z_b[0] = lower boundary, z_b[1] = upper boundary'

    h = np.zeros(2)
    x = None
    z = None

    if bc_hor[0] in ['periodic']:
        assert bc_hor[1] in ['periodic'], 'Periodic boundary conditions must be prescribed at both boundaries'
        x = np.linspace(x_b[0], x_b[1], N[0], endpoint=False)
        h[0] = x[1] - x[0]

    if bc_hor[0] in ['dirichlet', 'neumann']:
        x = np.linspace(x_b[0], x_b[1], N[0] + 2, endpoint=True)
        x = x[1 : N[0] + 1]
        h[0] = x[1] - x[0]

    if bc_ver[0] in ['periodic']:
        assert bc_ver[1] in ['periodic'], 'Periodic boundary conditions must be prescribed at both boundaries'
        z = np.linspace(z_b[0], z_b[1], N[1], endpoint=False)
        h[1] = z[1] - z[0]

    if bc_ver[0] in ['dirichlet', 'neumann']:
        z = np.linspace(z_b[0], z_b[1], N[1] + 2, endpoint=True)
        z = z[1 : N[1] + 1]
        h[1] = z[1] - z[0]

    xx, zz = np.meshgrid(x, z, indexing="ij")
    return xx, zz, h


#
#
#
def get2DMatrix(N, h, bc_hor, bc_ver, order):
    assert np.size(N) == 2, 'N needs to be an array with two entries: N[0]=Nx and N[1]=Nz'
    assert np.size(h) == 2, 'h needs to be an array with two entries: h[0]=dx and h[1]=dz'

    Ax = getMatrix(N[0], h[0], bc_hor[0], bc_hor[1], order)
    Az = getMatrix(N[1], h[1], bc_ver[0], bc_ver[1], order)

    Dx = sp.kron(Ax, sp.eye(N[1]), format="csr")
    Dz = sp.kron(sp.eye(N[0]), Az, format="csr")

    return Dx, Dz
