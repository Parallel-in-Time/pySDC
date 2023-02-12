import numpy as np

from pySDC.helpers.problem_helper import get_finite_difference_stencil


def get_finite_difference_eigenvalues(derivative, order, stencil_type=None, steps=None, dx=None, L=1.0):
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
        stencil_type (str): The type of stencil, i.e. 'forward', 'backward', or 'center'
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
    weights, offsets = get_finite_difference_stencil(derivative=derivative, order=order, stencil_type=stencil_type, steps=steps)

    # get the impact of the stencil in Fourier space
    for n in range(N):
        for i in range(len(weights)):
            eigenvalues[n] += weights[i] * np.exp(2 * np.pi * 1j * n / N * offsets[i]) * 1.0 / (dx**derivative)

    return eigenvalues
