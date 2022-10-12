import numpy as np
from scipy.special import factorial


def get_finite_difference_stencil(derivative, order, type=None, steps=None):
    """
    Derive general finite difference stencils from Taylor expansions
    """

    if steps is not None:
        n = len(steps)
    elif type == 'center':
        n = order + derivative - (derivative + 1) % 2 // 1
        steps = np.arange(n) - n // 2
    elif type == 'forward':
        n = order + derivative
        steps = np.arange(n)
    elif type == 'backward':
        n = order + derivative
        steps = -np.arange(n)
    else:
        raise ValueError(
            f'Stencil must be of type "center", "forward" or "backward", not {type}. If you want something\
else, you can also give specific steps.'
        )

    # make a matrix that contains the Taylor coefficients
    A = np.zeros((n, n))
    idx = np.arange(n)
    inv_facs = 1.0 / factorial(idx)
    for i in range(0, n):
        A[i, :] = steps ** idx[i] * inv_facs[i]

    # make a right hand side vector that is zero everywhere except at the position of the desired derivative
    sol = np.zeros(n)
    sol[derivative] = 1.0

    # solve the linear system for the finite difference coefficients
    coeff = np.linalg.solve(A, sol)

    return coeff, steps


def get_finite_difference_matrix(
    derivative, order, type=None, steps=None, dx=None, size=None, dim=None, bc=None, cupy=False
):
    """
    Build FD matrix from stencils, with boundary conditions
    """
    if cupy:
        import cupyx.scipy.sparse as sp
    else:
        import scipy.sparse as sp

    if order > 2 and bc != 'periodic':
        raise NotImplementedError('Higher order allowed only for periodic boundary conditions')

    # get stencil
    coeff, steps = get_finite_difference_stencil(derivative=derivative, order=order, type=type, steps=steps)

    if bc == 'dirichlet-zero':
        A_1d = sp.diags(coeff, steps, shape=(size, size), format='csc')
    elif bc == 'periodic':
        A_1d = 0 * sp.eye(size, format='csc')
        for i in steps:
            A_1d += coeff[i] * sp.eye(size, k=steps[i])
            if steps[i] > 0:
                A_1d += coeff[i] * sp.eye(size, k=-size + steps[i])
            if steps[i] < 0:
                A_1d += coeff[i] * sp.eye(size, k=size + steps[i])
    else:
        raise NotImplementedError(f'Boundary conditons {bc} not implemented.')

    if dim == 1:
        A = A_1d
    elif dim == 2:
        A = sp.kron(A_1d, sp.eye(size)) + sp.kron(sp.eye(size), A_1d)
    elif dim == 3:
        A = (
            sp.kron(A_1d, sp.eye(size**2))
            + sp.kron(sp.eye(size**2), A_1d)
            + sp.kron(sp.kron(sp.eye(size), A_1d), sp.eye(size))
        )
    else:
        raise NotImplementedError(f'Dimension {dim} not implemented.')

    A /= dx**derivative

    return A
