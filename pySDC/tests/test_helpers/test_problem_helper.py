from pySDC.helpers.problem_helper import get_finite_difference_stencil
import pytest
import numpy as np


def fd_stencil_single(derivative, order, stencil_type):
    """
    Make a single tests where we generate a finite difference stencil using the generic framework above and compare to
    harscoded stencils that were implemented in a previous version of the code.

    Args:
        derivative (int): Order of the derivative
        order (int): Order of accuracy
        stencil_type (str): Type of the stencil

    Returns:
        None
    """
    if derivative == 1:
        if stencil_type == 'center':
            if order == 2:
                stencil = [-1.0, 0.0, 1.0]
                zero_pos = 2
                coeff = 1.0 / 2.0
            elif order == 4:
                stencil = [1.0, -8.0, 0.0, 8.0, -1.0]
                zero_pos = 3
                coeff = 1.0 / 12.0
            elif order == 6:
                stencil = [-1.0, 9.0, -45.0, 0.0, 45.0, -9.0, 1.0]
                zero_pos = 4
                coeff = 1.0 / 60.0
            else:
                raise NotImplementedError("Order " + str(order) + " not implemented.")
        elif stencil_type == 'upwind':
            if order == 1:
                stencil = [-1.0, 1.0]
                coeff = 1.0
                zero_pos = 2

            elif order == 2:
                stencil = [1.0, -4.0, 3.0]
                coeff = 1.0 / 2.0
                zero_pos = 3

            elif order == 3:
                stencil = [1.0, -6.0, 3.0, 2.0]
                coeff = 1.0 / 6.0
                zero_pos = 3

            elif order == 4:
                stencil = [-5.0, 30.0, -90.0, 50.0, 15.0]
                coeff = 1.0 / 60.0
                zero_pos = 4

            elif order == 5:
                stencil = [3.0, -20.0, 60.0, -120.0, 65.0, 12.0]
                coeff = 1.0 / 60.0
                zero_pos = 5
            else:
                raise NotImplementedError("Order " + str(order) + " not implemented.")
        else:
            raise NotImplementedError(
                f"No reference values for stencil_type \"{stencil_type}\" implemented for 1st derivative"
            )
    elif derivative == 2:
        if stencil_type == 'center':
            coeff = 1.0
            if order == 2:
                stencil = [1, -2, 1]
                zero_pos = 2
            elif order == 4:
                stencil = [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]
                zero_pos = 3
            elif order == 6:
                stencil = [1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90]
                zero_pos = 4
            elif order == 8:
                stencil = [-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560]
                zero_pos = 5
        else:
            raise NotImplementedError(
                f"No reference values for stencil_type \"{stencil_type}\" implemented for 2nd derivative"
            )
    else:
        raise NotImplementedError(f"No reference values for derivative {derivative} implemented")

    # convert the reference values to a common way of writing with what we generate here
    coeff_reference = np.array(stencil) * coeff
    steps_reference = np.append(np.arange(-zero_pos + 1, 1), np.arange(1, zero_pos))[: len(coeff_reference)]
    sorted_idx_reference = np.argsort(steps_reference)

    coeff, steps = get_finite_difference_stencil(derivative=derivative, order=order, stencil_type=stencil_type)
    sorted_idx = np.argsort(steps)
    assert np.allclose(
        coeff_reference[sorted_idx_reference], coeff[sorted_idx]
    ), f"Got different FD coefficients for derivative {derivative} with order {order} and stencil_type {stencil_type}! Expected {coeff_reference[sorted_idx_reference]}, got {coeff[sorted_idx]}."

    assert np.allclose(
        steps_reference[sorted_idx_reference], steps[sorted_idx]
    ), f"Got different FD offsets for derivative {derivative} with order {order} and stencil_type {stencil_type}! Expected {steps_reference[sorted_idx_reference]}, got {steps[sorted_idx]}."


@pytest.mark.base
def test_fd_stencils():
    """
    Perform multiple tests for the generic FD stencil generating framework.

    Returns:
        None
    """
    # Make tests to things that were previously implemented in the code
    for order in [1, 2, 3, 4, 5]:
        fd_stencil_single(1, order, 'upwind')
    for order in [2, 4, 6]:
        fd_stencil_single(1, order, 'center')
    for order in [2, 4, 6, 8]:
        fd_stencil_single(2, order, 'center')

    # Make some tests comparing to Wikipedia at https://en.wikipedia.org/wiki/Finite_difference_coefficient
    coeff, steps = get_finite_difference_stencil(derivative=1, order=3, stencil_type='forward')
    expect_coeff = [-11.0 / 6.0, 3.0, -3.0 / 2.0, 1.0 / 3.0]
    assert np.allclose(
        coeff, expect_coeff
    ), f"Error in thrid order forward stencil for 1st derivative! Expected {expect_coeff}, got {coeff}."

    coeff, steps = get_finite_difference_stencil(derivative=2, order=2, stencil_type='backward')
    expect_coeff = [-1, 4, -5, 2][::-1]
    assert np.allclose(
        coeff, expect_coeff
    ), f"Error in second order backward stencil for 2nd derivative! Expected {expect_coeff}, got {coeff}."

    # test if we get the correct result when we put in steps rather than a stencil_type
    new_coeff, _ = get_finite_difference_stencil(derivative=2, order=2, steps=steps)
    assert np.allclose(coeff, new_coeff), f"Error when setting steps yourself! Expected {expect_coeff}, got {coeff}."
