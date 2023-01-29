import pytest


@pytest.mark.base
@pytest.mark.parametrize("equation", ['heat', 'advection'])
def test_FD_eigenvalues(equation):
    """
    Test two particular special cases of computing eigenvalues of a finite difference discretization.
    """
    import numpy as np
    from pySDC.projects.Resilience.FDeigenvalues import get_finite_difference_eigenvalues

    order = 2
    type = 'center'
    dx = 0.1
    L = 1.0
    N = int(L // dx)
    n = np.arange(N, dtype=complex)

    if equation == 'heat':
        derivative = 2
        expect = -2.0 / (dx**2.0) * (1.0 - np.cos(2 * np.pi * n / N))
    elif equation == 'advection':
        derivative = 1
        expect = 1.0j / (dx) * (np.sin(2 * np.pi * n / N))

    assert np.allclose(
        expect, get_finite_difference_eigenvalues(derivative, order, type, dx=dx, L=L)
    ), f"Error when doing {equation}"


if __name__ == '__main__':
    for equation in ['heat', 'advection']:
        test_FD_eigenvalues(equation)
