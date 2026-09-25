import scipy.sparse as sp

from pySDC.helpers.problem_helper import (
    get_finite_difference_matrix,
    get_finite_difference_stencil,
)


def getUpwindMatrix(N, dx, order):
    A, _ = get_finite_difference_matrix(
        derivative=1, order=order, stencil_type='upwind', dx=dx, size=N, dim=1, bc='periodic'
    )
    return sp.csc_matrix(A)


def getMatrix(N, dx, bc_left, bc_right, order):
    r"""
    Centered first derivative matrix for homogeneous boundary conditions.

    For periodic boundaries this is just the generic finite difference matrix. For Neumann and
    Dirichlet boundaries only the interior stencil is taken from the generic helper, while the
    rows next to the boundary keep the closures that were derived for this problem.

    These closures are *not* interchangeable with the ones the generic helper builds, even though
    the generic ones are more accurate: the Boussinesq operator differentiates the pressure with
    the Neumann closure and the vertical velocity with the Dirichlet closure, and neutral stability
    of the resulting wave operator depends on how those two closures relate to each other. Building
    each of them in isolation, as the generic helper does, gives a discretisation whose spectrum has
    positive real parts and which therefore grows exponentially in time. See #233.
    """
    assert bc_left in ['periodic', 'neumann', 'dirichlet'], "Unknown type of BC"
    assert bc_right in ['periodic', 'neumann', 'dirichlet'], "Unknown type of BC"

    if bc_left == 'periodic' or bc_right == 'periodic':
        assert bc_left == bc_right, "Periodic BC can only be selected for both sides simultaneously"
        A, _ = get_finite_difference_matrix(
            derivative=1, order=order, stencil_type='center', dx=dx, size=N, dim=1, bc='periodic'
        )
        return sp.csc_matrix(A)

    assert order in [2, 4], "Neumann and Dirichlet closures are only available for order 2 and 4"

    coeff, steps = get_finite_difference_stencil(derivative=1, order=order, stencil_type='center')
    A = sp.lil_matrix(sp.diags(coeff, steps, shape=(N, N)))

    # Neumann boundary conditions
    if bc_left == 'neumann':
        A[0, :] = 0.0
        if order == 2:
            A[0, 0] = -2.0 / 3.0
            A[0, 1] = 2.0 / 3.0
        elif order == 4:
            A[0, 0] = -2.0 / 3.0
            A[0, 1] = 2.0 / 3.0
            A[1, 0] = -5.0 / 9.0
            A[1, 1] = -1.0 / 36.0

    if bc_right == 'neumann':
        A[N - 1, :] = 0.0
        if order == 2:
            A[N - 1, N - 2] = -2.0 / 3.0
            A[N - 1, N - 1] = 2.0 / 3.0
        elif order == 4:
            A[N - 2, N - 1] = 5.0 / 9.0
            A[N - 2, N - 2] = 1.0 / 36.0
            A[N - 1, N - 1] = 2.0 / 3.0
            A[N - 1, N - 2] = -2.0 / 3.0

    # Dirichlet boundary conditions. For order 2 the ghost value drops out of the stencil, so
    # there is nothing to do.
    if bc_left == 'dirichlet' and order == 4:
        A[0, :] = 0.0
        A[0, 1] = 1.0 / 2.0

    if bc_right == 'dirichlet' and order == 4:
        A[N - 1, :] = 0.0
        A[N - 1, N - 2] = -1.0 / 2.0

    return sp.csc_matrix(A / dx)
