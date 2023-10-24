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
    derivative,
    order,
    stencil_type=None,
    steps=None,
    dx=None,
    size=None,
    dim=None,
    bc=None,
    cupy=False,
    bc_params=None,
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
        numpy.ndarray: Vector containing information about the boundary conditions
    """
    if cupy:
        import cupyx.scipy.sparse as sp
    else:
        import scipy.sparse as sp

    # get stencil
    coeff, steps = get_finite_difference_stencil(
        derivative=derivative, order=order, stencil_type=stencil_type, steps=steps
    )

    if type(bc) is not tuple:
        assert type(bc) == str, 'Please pass BCs as string or tuple of strings'
        bc = (bc, bc)
    bc_params = bc_params if bc_params is not None else {}
    if type(bc_params) is not list:
        bc_params = [bc_params, bc_params]

    b = np.zeros(size**dim)

    if bc[0] == 'periodic' and bc[1] == 'periodic':
        A_1d = 0 * sp.eye(size, format='csc')
        for i in steps:
            A_1d += coeff[i] * sp.eye(size, k=steps[i])
            if steps[i] > 0:
                A_1d += coeff[i] * sp.eye(size, k=-size + steps[i])
            if steps[i] < 0:
                A_1d += coeff[i] * sp.eye(size, k=size + steps[i])
    else:
        A_1d = sp.diags(coeff, steps, shape=(size, size), format='lil')

        bc_params_defaults = {
            'val': 0.0,
            'neumann_bc_order': order,
            'reduce': False,
        }
        bc_params[0] = {**bc_params_defaults, **bc_params[0]}
        bc_params[1] = {**bc_params_defaults, **bc_params[1]}

        if 'dirichlet' in bc[0]:
            if bc_params[0]['reduce']:
                for i in range(0, abs(min(steps))):
                    b_coeff, b_steps = get_finite_difference_stencil(
                        derivative=derivative,
                        order=2 * (i + 1),
                        stencil_type='center',
                    )
                    print(A_1d.toarray())
                    A_1d[i, :] = 0
                    A_1d[i, : len(b_coeff) - 1] = b_coeff[1:]
                    b[i] = bc_params[0]['val'] * b_coeff[0] / dx**derivative
                    print(A_1d.toarray())
                    print(i, b_coeff, b_steps)
            else:
                for i in range(0, abs(min(steps))):
                    b_steps = np.arange(-(i + 1), order + derivative - (i + 1))
                    b_coeff, b_steps = get_finite_difference_stencil(derivative=derivative, order=order, steps=b_steps)
                    A_1d[i, : len(b_coeff) - 1] = b_coeff[1:]
                    b[i] = bc_params[0]['val'] * b_coeff[0] / dx**derivative
        if 'dirichlet' in bc[1]:
            if bc_params[1]['reduce']:
                for i in range(0, abs(min(steps))):
                    b_coeff, b_steps = get_finite_difference_stencil(
                        derivative=derivative,
                        order=2 * (i + 1),
                        stencil_type='center',
                    )
                    A_1d[-i - 1, -len(b_coeff) + 1 :] = b_coeff[:-1]
                    b[-i - 1] = bc_params[1]['val'] * b_coeff[-1] / dx**derivative
            else:
                for i in range(0, abs(max(steps))):
                    b_steps = np.arange(-(order + derivative) + (i + 2), (i + 2))
                    b_coeff, b_steps = get_finite_difference_stencil(derivative=derivative, order=order, steps=b_steps)
                    A_1d[-i - 1, -len(b_coeff) + 1 :] = b_coeff[:-1]
                    b[-i - 1] = bc_params[1]['val'] * b_coeff[-1] / dx**derivative
        if 'neumann' in bc[0]:
            # generate the one-sided stencil to discretize the first derivative at the boundary
            bc_coeff_left, bc_steps_left = get_finite_difference_stencil(
                derivative=1, order=bc_params[0]['neumann_bc_order'], stencil_type='forward'
            )

            # check if we can just use the available stencil or if we need to generate a new one
            if steps.min() == -1:
                coeff_left = coeff.copy()
            else:  # need to generate lopsided stencils
                raise NotImplementedError(
                    'Neumann BCs on the right are not implemented for your desired stencil. Maybe try a lower order'
                )

            # modify system matrix and inhomogeneity according to BC
            b[0] = bc_params[0]['val'] * (coeff_left[0] / dx**derivative) / (bc_coeff_left[0] / dx)
            A_1d[0, : len(bc_coeff_left) - 1] -= coeff_left[0] / bc_coeff_left[0] * bc_coeff_left[1:]
        if 'neumann' in bc[1]:
            # generate the one-sided stencil to discretize the first derivative at the boundary
            bc_coeff_right, bc_steps_right = get_finite_difference_stencil(
                derivative=1, order=bc_params[1]['neumann_bc_order'], stencil_type='backward'
            )

            # check if we can just use the available stencil or if we need to generate a new one
            if steps.max() == +1:
                coeff_right = coeff.copy()
            else:  # need to generate lopsided stencils
                raise NotImplementedError(
                    'Neumann BCs on the right are not implemented for your desired stencil. Maybe try a lower order'
                )

            # modify system matrix and inhomogeneity according to BC
            b[-1] = bc_params[1]['val'] * (coeff_right[-1] / dx**derivative) / (bc_coeff_right[0] / dx)
            A_1d[-1, -len(bc_coeff_right) + 1 :] -= coeff_right[-1] / bc_coeff_right[0] * bc_coeff_right[::-1][:-1]

    print(A_1d.toarray())
    print(b)
    # elif "dirichlet" in bc:
    # elif "neumann" in bc:
    #     """
    #     We will solve only for values within the boundary because the values on the boundary can be determined from the
    #     discretization of the boundary condition (BC). Therefore, we discretize both the BC and the original problem
    #     such that the stencils reach into the boundary with one element. We then proceed to eliminate the values on the
    #     boundary, which modifies the finite difference matrix and yields an inhomogeneity if the BCs are inhomogeneous.

    #     Keep in mind that centered stencils are often more efficient in terms of sparsity of the resulting matrix. High
    #     order centered discretizations will reach beyond the boundary and hence need to be replaced with lopsided
    #     stencils near the boundary, changing the number of non-zero diagonals.
    #     """
    #     bc_params_defaults = {
    #             'val_left': 0.,
    #             'val_right': 0.,
    #             'neumann_bc_order': order,
    #             'reduce': False,
    #     }
    #     bc_params[0] = {**bc_params_defaults, **bc_params[0]} if 'neumann' in bc[0] else bc_params[0]
    #     bc_params[1] = {**bc_params_defaults, **bc_params[1]} if 'neumann' in bc[1] else bc_params[1]
    #
    #     # check if we need to alter the sparsity structure because of the BC
    #     if steps.min() < -1 or steps.max() > 1:
    #         A_1d = sp.diags(coeff, steps, shape=(size, size), format='lil')
    #     else:
    #         A_1d = sp.diags(coeff, steps, shape=(size, size), format='csc')

    #     if derivative == 1 and (steps.min() < -1 or steps.max() > 1):
    #         neumann_bc_order = neumann_bc_order if neumann_bc_order else order + 1
    #         assert neumann_bc_order != order, 'Need different stencils for BC and the rest'
    #     else:
    #         neumann_bc_order = neumann_bc_order if neumann_bc_order else order

    #     if dim > 1 and (val_left != 0.0 or val_right != 0):
    #         raise NotImplementedError(
    #             f'Non-zero Neumann BCs are only implemented in 1D. You asked for {dim} dimensions.'
    #         )

    #     # ---- left boundary ----
    #     # generate the one-sided stencil to discretize the first derivative at the boundary
    #     if 'neumann' in bc[0]:
    #     bc_coeff_left, bc_steps_left = get_finite_difference_stencil(
    #         derivative=1, order=neumann_bc_order, stencil_type='forward'
    #     )

    #     # check if we can just use the available stencil or if we need to generate a new one
    #     if steps.min() == -1:
    #         coeff_left = coeff.copy()
    #     else:  # need to generate lopsided stencils
    #         raise NotImplementedError(
    #             'Neumann BCs on the right are not implemented for your desired stencil. Maybe try a lower order'
    #         )

    #     # modify system matrix and inhomogeneity according to BC
    #     b[0] = val_left * (coeff_left[0] / dx**derivative) / (bc_coeff_left[0] / dx)
    #     A_1d[0, : len(bc_coeff_left) - 1] -= coeff_left[0] / bc_coeff_left[0] * bc_coeff_left[1:]

    #     # ---- right boundary ----
    #     # generate the one-sided stencil to discretize the first derivative at the boundary
    #     bc_coeff_right, bc_steps_right = get_finite_difference_stencil(
    #         derivative=1, order=neumann_bc_order, stencil_type='backward'
    #     )

    #     # check if we can just use the available stencil or if we need to generate a new one
    #     if steps.max() == +1:
    #         coeff_right = coeff.copy()
    #     else:  # need to generate lopsided stencils
    #         raise NotImplementedError(
    #             'Neumann BCs on the right are not implemented for your desired stencil. Maybe try a lower order'
    #         )

    #     # modify system matrix and inhomogeneity according to BC
    #     b[-1] = val_right * (coeff_right[-1] / dx**derivative) / (bc_coeff_right[0] / dx)
    #     A_1d[-1, -len(bc_coeff_right) + 1 :] -= coeff_right[-1] / bc_coeff_right[0] * bc_coeff_right[::-1][:-1]
    # else:
    #     raise NotImplementedError(f'Boundary conditions \"{bc}\" not implemented.')

    # TODO: extend the BCs to higher dimensions
    A_1d = A_1d.tocsc()
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

    return A, b


def get_1d_grid(size, bc, left_boundary=0.0, right_boundary=1.0):
    """
    Generate a grid in one dimension and obtain mesh spacing for finite difference discretization.

    Args:
        size (int): Number of degrees of freedom per dimension
        bc (str): Boundary conditions for both sides
        left_boundary (float): x value at the left boundary
        right_boundary (float): x value at the right boundary

    Returns:
        float: mesh spacing
        numpy.ndarray: 1d mesh
    """
    L = right_boundary - left_boundary
    if bc == 'periodic':
        dx = L / size
        xvalues = np.array([left_boundary + dx * i for i in range(size)])
    elif "dirichlet" in bc or "neumann" in bc:
        dx = L / (size + 1)
        xvalues = np.array([left_boundary + dx * (i + 1) for i in range(size)])
    else:
        raise NotImplementedError(f'Boundary conditions \"{bc}\" not implemented.')

    return dx, xvalues
