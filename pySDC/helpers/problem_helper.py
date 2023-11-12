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


def get_finite_difference_stencil(derivative, order=None, stencil_type=None, steps=None):
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

    # sort coefficients and steps
    coeff = coeff[np.argsort(steps)]
    steps = np.sort(steps)

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

    if bc[0] == 'periodic':
        assert bc[1] == 'periodic'
        A_1d = 0 * sp.eye(size, format='csc')
        for i in steps:
            A_1d += coeff[i] * sp.eye(size, k=steps[i])
            if steps[i] > 0:
                A_1d += coeff[i] * sp.eye(size, k=-size + steps[i])
            if steps[i] < 0:
                A_1d += coeff[i] * sp.eye(size, k=size + steps[i])
    else:
        A_1d = sp.diags(coeff, steps, shape=(size, size), format='lil')

        # Default parameters for Dirichlet and Neumann BCs
        bc_params_defaults = {
            'val': 0.0,
            'neumann_bc_order': order,
            'reduce': False,
        }

        # Loop over each side (0 for left, 1 for right)
        for iS in [0, 1]:
            # -- check Boundary condition types
            assert "neumann" in bc[iS] or "dirichlet" in bc[iS], f"unknown BC type : {bc[iS]}"

            # -- boundary condition parameters
            bc_params[iS] = {**bc_params_defaults, **bc_params[iS]}
            par = bc_params[iS]

            # -- extract parameters and raise an error if additionals
            val = par.pop('val')
            reduce = par.pop('reduce')
            neumann_bc_order = par.pop('neumann_bc_order')
            assert len(par) == 0, f"unused BCs parameters : {par}"

            # -- half stencil width
            sWidth = -min(steps) if iS == 0 else max(steps)

            # -- loop over lines of A that have to be modified
            for i in range(sWidth):
                # -- index of the line
                iLine = i if iS == 0 else -i - 1
                # -- slice for coefficients used in the A matrix
                sCoeff = slice(1, None) if iS == 0 else slice(None, -1)
                # -- index of coefficient used in the b vector
                iCoeff = 0 if iS == 0 else -1

                if reduce:
                    # -- reduce order close to boundary
                    b_coeff, b_steps = get_finite_difference_stencil(
                        derivative=derivative,
                        order=2 * (i + 1),
                        stencil_type='center',
                    )
                else:
                    # -- shift stencil close to boundary
                    b_steps = (
                        np.arange(-(i + 1), order + derivative - (i + 1))
                        if iS == 0
                        else np.arange(-(order + derivative) + (i + 2), (i + 2))
                    )

                    b_coeff, b_steps = get_finite_difference_stencil(derivative=derivative, steps=b_steps)

                # -- column slice where to put coefficients in the A matrix
                colSlice = slice(None, len(b_coeff) - 1) if iS == 0 else slice(-len(b_coeff) + 1, None)

                # -- modify A
                A_1d[iLine, :] = 0
                A_1d[iLine, colSlice] = b_coeff[sCoeff]

                if "dirichlet" in bc[iS]:
                    # -- modify b
                    b[iLine] = val * b_coeff[iCoeff]

                elif "neumann" in bc[iS]:
                    nOrder = neumann_bc_order

                    # -- generate the first derivative stencil
                    n_coeff, n_steps = get_finite_difference_stencil(
                        derivative=1, order=nOrder, stencil_type="forward" if iS == 0 else "backward"
                    )

                    # -- column slice where to put coefficients in the A matrix
                    colSlice = slice(None, len(n_coeff) - 1) if iS == 0 else slice(-len(n_coeff) + 1, None)

                    # -- additional modification to A
                    A_1d[iLine, colSlice] -= b_coeff[iCoeff] / n_coeff[iCoeff] * n_coeff[sCoeff]

                    # -- modify B
                    b[iLine] = val * b_coeff[iCoeff] / n_coeff[iCoeff] * dx

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
    b /= dx**derivative

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
