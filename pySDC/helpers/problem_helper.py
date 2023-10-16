import numpy as np
from scipy.special import factorial


def get_steps(derivative, order, stencil_type):
    """
    Get the offsets for the FD stencil.

    Args:
        derivative (int): Order of the derivative
        order (int): Order of accuracy
        stencil_type (str): Type of the stencil
        steps (list): Provide specific steps, overrides `stencil_type`

    Returns:
        int: The number of elements in the stencil
        numpy.ndarray: The offsets for the stencil
    """
    if stencil_type == 'center':
        n = order + derivative - (derivative + 1) % 2 // 1
        steps = np.arange(n) - n // 2
    elif stencil_type == 'forward':
        n = order + derivative
        steps = np.arange(n)
    elif stencil_type == 'backward':
        n = order + derivative
        steps = -np.arange(n)
    elif stencil_type == 'upwind':
        n = order + derivative

        if n <= 3:
            n, steps = get_steps(derivative, order, 'backward')
        else:
            steps = np.append(-np.arange(n - 1)[::-1], [1])
    else:
        raise ValueError(
            f'Stencil must be of type "center", "forward", "backward" or "upwind", not {stencil_type}. If you want something else you can also give specific steps.'
        )
    return n, steps


def get_finite_difference_stencil(derivative, order, stencil_type=None, steps=None):
    """
    Derive general finite difference stencils from Taylor expansions

    Args:
        derivative (int): Order of the derivative
        order (int): Order of accuracy
        stencil_type (str): Type of the stencil
        steps (list): Provide specific steps, overrides `stencil_type`

    Returns:
        numpy.ndarray: The weights of the stencil
        numpy.ndarray: The offsets for the stencil
    """

    if steps is not None:
        n = len(steps)
    else:
        n, steps = get_steps(derivative, order, stencil_type)

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
    derivative, order, stencil_type=None, steps=None, dx=None, size=None, dim=None, bc=None, cupy=False
):
    """
    Build FD matrix from stencils, with boundary conditions.
    Keep in mind that the boundary conditions may require further modification of the right hand side.

    Args:
        derivative (int): Order of the spatial derivative
        order (int): Order of accuracy
        stencil_type (str): Type of stencil
        steps (list): Provide specific steps, overrides `stencil_type`
        dx (float): Mesh width
        size (int): Number of degrees of freedom per dimension
        dim (int): Number of dimensions
        bc (str): Boundary conditions for both sides
        cupy (bool): Construct a GPU ready matrix if yes

    Returns:
        Sparse matrix: Finite difference matrix
    """
    if cupy:
        import cupyx.scipy.sparse as sp
    else:
        import scipy.sparse as sp

    # get stencil
    coeff, steps = get_finite_difference_stencil(
        derivative=derivative, order=order, stencil_type=stencil_type, steps=steps
    )

    if "dirichlet" in bc:
        A_1d = sp.diags(coeff, steps, shape=(size, size), format='csc')
    elif "neumann" in bc:
        A_1d = sp.diags(coeff, steps, shape=(size, size), format='lil')

        # replace the first and last lines with one-sided stencils for a first derivative of the same order as the rest
        bc_coeff, bc_steps = get_finite_difference_stencil(derivative=1, order=order, stencil_type='forward')
        A_1d[0, :] = 0.0
        A_1d[0, bc_steps] = bc_coeff * (dx ** (derivative - 1))
        A_1d[-1, :] = 0.0
        A_1d[-1, -bc_steps - 1] = bc_coeff * (dx ** (derivative - 1))
    elif bc == 'periodic':
        A_1d = 0 * sp.eye(size, format='csc')
        for i in steps:
            A_1d += coeff[i] * sp.eye(size, k=steps[i])
            if steps[i] > 0:
                A_1d += coeff[i] * sp.eye(size, k=-size + steps[i])
            if steps[i] < 0:
                A_1d += coeff[i] * sp.eye(size, k=size + steps[i])
    else:
        raise NotImplementedError(f'Boundary conditions {bc} not implemented.')

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
