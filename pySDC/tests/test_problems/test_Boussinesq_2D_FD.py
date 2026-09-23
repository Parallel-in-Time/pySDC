import pytest


@pytest.mark.base
@pytest.mark.parametrize('order', [2, 4])
@pytest.mark.parametrize('bc', ['periodic', 'neumann', 'dirichlet'])
def test_first_derivative(order, bc):
    """The matrices differentiate a function that satisfies the boundary conditions."""
    import numpy as np
    from pySDC.implementations.problem_classes.boussinesq_helpers.buildFDMatrix import getMatrix

    L, N = 1.0, 64
    if bc == 'periodic':
        x = np.linspace(0.0, L, N, endpoint=False)
        u, du = np.sin(2 * np.pi * x), 2 * np.pi * np.cos(2 * np.pi * x)
    else:
        x = np.linspace(0.0, L, N + 2)[1 : N + 1]
        # homogeneous Dirichlet and Neumann data, respectively
        u = np.sin(np.pi * x) if bc == 'dirichlet' else np.cos(np.pi * x)
        du = np.pi * np.cos(np.pi * x) if bc == 'dirichlet' else -np.pi * np.sin(np.pi * x)

    dx = x[1] - x[0]
    error = np.linalg.norm(getMatrix(N, dx, bc, bc, order).dot(u) - du, np.inf)
    assert error < 5e-2, f'Derivative is off by {error:.3e} for {bc} BCs of order {order}'


@pytest.mark.base
def test_upwind_matrix():
    """The upwind matrix differentiates a periodic function."""
    import numpy as np
    from pySDC.implementations.problem_classes.boussinesq_helpers.buildFDMatrix import getUpwindMatrix

    N = 64
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    dx = x[1] - x[0]
    for order in [1, 2, 3, 4, 5]:
        error = np.linalg.norm(
            getUpwindMatrix(N, dx, order).dot(np.sin(2 * np.pi * x)) - 2 * np.pi * np.cos(2 * np.pi * x), np.inf
        )
        assert error < 5e-1, f'Upwind derivative of order {order} is off by {error:.3e}'


@pytest.mark.base
def test_operator_is_neutrally_stable():
    """
    The Boussinesq operator must not have eigenvalues with a positive real part.

    This is what ties the problem to the boundary closures in `boussinesq_helpers.buildFDMatrix`
    rather than to the ones `pySDC.helpers.problem_helper` builds: the vertical derivative of the
    pressure uses the Neumann closure and the one of the vertical velocity uses the Dirichlet
    closure, and the two have to fit together. Closures that are each accurate on their own, but
    not compatible with one another, give an operator that grows exponentially in time. See #233.
    """
    import numpy as np
    from pySDC.implementations.problem_classes.Boussinesq_2D_FD_imex import boussinesq_2d_imex

    prob = boussinesq_2d_imex(nvars=(4, 12, 10))
    eigenvalues = np.linalg.eigvals(prob.M.toarray())
    growth = eigenvalues.real.max()
    assert growth < 1e-10 * np.abs(eigenvalues).max(), f'Boussinesq operator grows at rate {growth:.3e}'
