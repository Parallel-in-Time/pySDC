import numpy as np

from pySDC.helpers.problem_helper import get_finite_difference_stencil


def get_finite_difference_eigenvalues(derivative, order, type=None, steps=None, dx=None, L=1.0):
    """
    Compute the eigenvalues of the finite difference (FD) discretization using Fourier transform.

    In Fourier space, the offsets in the FD discretizations manifest as multiplications by
        exp(2 * pi * j * n / N * offset).
    Then, all you need to do is sum up the contributions from all entries in the stencil and Bob's your uncle,
    you have computed the eigenvalues.

    There are going to be as many eigenvalues as there are space elements.
    Please be aware that these are in general complex.

    Args:
        derivative (int): The order of the derivative
        order (int): The order of accuracy of the derivative
        type (str): The type of stencil, i.e. 'forward', 'backward', or 'center'
        steps (list): If you want an exotic stencil like upwind, you can give the offsets here
        dx (float): The mesh spacing
        L (float): The length of the interval in space

    Returns:
        numpy.ndarray: The complex (!) eigenvalues.
    """
    # prepare variables
    N = int(L // dx)
    eigenvalues = np.zeros(N, dtype=complex)

    # get the stencil
    weights, offsets = get_finite_difference_stencil(derivative=derivative, order=order, type=type, steps=steps)

    # get the impact of the stencil in Fourier space
    for n in range(N):
        for i in range(len(weights)):
            eigenvalues[n] += weights[i] * np.exp(2 * np.pi * 1j * n / N * offsets[i]) * 1.0 / (dx**derivative)

    return eigenvalues


def test():
    """
    Test a particular special case
    """
    order = 2
    type = 'center'
    dx = 0.1
    L = 1.0
    N = int(L // dx)
    n = np.arange(N, dtype=complex)

    # heat
    derivative = 2
    expect = -2.0 / (dx**2.0) * (1.0 - np.cos(2 * np.pi * n / N))
    assert np.allclose(
        expect, get_finite_difference_eigenvalues(derivative, order, type, dx=dx, L=L)
    ), "Error when doing heat"

    # advection
    derivative = 1
    expect = 1.0j / (dx) * (np.sin(2 * np.pi * n / N))
    assert np.allclose(
        expect, get_finite_difference_eigenvalues(derivative, order, type, dx=dx, L=L)
    ), "Error when doing advection"


if __name__ == '__main__':
    test()
