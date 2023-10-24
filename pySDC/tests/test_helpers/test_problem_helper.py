from pySDC.helpers.problem_helper import get_finite_difference_stencil
import pytest
import numpy as np


def fd_stencil_single(derivative, order, stencil_type):
    """
    Make a single tests where we generate a finite difference stencil using the generic framework above and compare to
    hardcoded stencils that were implemented in a previous version of the code.

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


@pytest.mark.base
@pytest.mark.parametrize('bc_left', [0.0, 7.0])
@pytest.mark.parametrize('bc_right', [0.0, 9.0])
@pytest.mark.parametrize('dx', [0.1, 10.0])
@pytest.mark.parametrize('derivative', [1, 2])
@pytest.mark.parametrize('order', [2])
def test_Neumann_bcs(derivative, bc_left, bc_right, dx, order):
    from pySDC.helpers.problem_helper import get_finite_difference_matrix

    A, b = get_finite_difference_matrix(
        derivative=derivative,
        order=order,
        stencil_type='center',
        bc='neumann',
        dim=1,
        size=4,
        dx=dx,
        bc_params=[{'val': bc_left}, {'val': bc_right}],
    )

    if derivative == 1:
        expect = np.zeros(A.shape[0])
        expect[0] = -2 / (3 * dx)
        expect[1] = +2 / (3 * dx)
        assert np.allclose(
            A.toarray()[0, :], expect
        ), f'Error in left boundary, expected {expect} got {A.toarray()[0, :]}!'
        expect = np.zeros(A.shape[0])
        expect[-2] = -2 / (3 * dx)
        expect[-1] = +2 / (3 * dx)
        assert np.allclose(
            A.toarray()[-1, :], expect
        ), f'Error in right boundary, expected {expect} got {A.toarray()[-1, :]}!'

        assert np.isclose(
            b[-1], bc_right / 3.0
        ), f'Error in right boundary value! Expected {bc_right / 3.}, got {b[-1]}'
        assert np.isclose(b[0], bc_left / 3.0), f'Error in left boundary value! Expected {bc_left / 3}, got {b[0]}'

    if derivative == 2:
        expect = np.zeros(A.shape[0])
        expect[0] = -2 / (3 * dx**2)
        expect[1] = +2 / (3 * dx**2)
        assert np.allclose(
            A.toarray()[0, :], expect
        ), f'Error in left boundary, expected {expect} got {A.toarray()[0, :]}!'
        assert np.allclose(
            A.toarray()[-1, :], expect[::-1]
        ), f'Error in right boundary, expected {expect[::-1]} got {A.toarray()[-1, :]}!'

        assert np.isclose(
            b[-1], bc_right * 2 / (3 * dx)
        ), f'Error in right boundary value! Expected {bc_right * 2 / (3*dx)}, got {b[-1]}'
        assert np.isclose(
            b[0], -bc_left * 2 / (3 * dx)
        ), f'Error in left boundary value! Expected {-bc_left * 2 / (3*dx)}, got {b[0]}'


@pytest.mark.parametrize('size', [10, 20, 50])
@pytest.mark.parametrize('order', [2, 4, 6, 8])
def test_Dirichtlet_bcs(order, size):
    from pySDC.helpers.problem_helper import get_finite_difference_matrix, get_1d_grid
    from scipy.sparse.linalg import spsolve
    from numpy.random import rand
    import numpy as np

    L = 2 * np.pi

    dx, x = get_1d_grid(size, 'dirichlet', 0, L)

    bc_right = rand()
    bc_left = rand()

    A, b = get_finite_difference_matrix(
        derivative=2,
        order=order,
        stencil_type='center',
        bc='dirichlet',
        dim=1,
        size=size,
        dx=dx,
        bc_params=[{'val': bc_left}, {'val': bc_right}],
    )

    u = spsolve(A, -b)

    u_expect = (bc_right - bc_left) * x / L + bc_left
    assert np.allclose(u, u_expect), 'Dirichlet BCs failed!'


@pytest.mark.parametrize('order', [2, 4, 6, 8])
def test_Dirichtlet_bcs_sin(order):
    from pySDC.helpers.problem_helper import get_finite_difference_matrix, get_1d_grid
    from scipy.sparse.linalg import spsolve
    from numpy.random import rand
    import numpy as np

    L = 2 * np.pi
    # L = 7.
    bc_right = rand()
    bc_left = rand()
    k = 4.0
    reduce = True

    def u_num(size):
        dx, x = get_1d_grid(size, 'dirichlet', 0, L)

        A, b = get_finite_difference_matrix(
            derivative=2,
            order=order,
            stencil_type='center',
            bc='dirichlet',
            dim=1,
            size=size,
            dx=dx,
            bc_params=[{'val': bc_left, 'reduce': reduce}, {'val': bc_right, 'reduce': reduce}],
        )
        print(np.linalg.cond(A.toarray()))

        source_term = np.sin(k * x)

        u = spsolve(A, source_term - b)

        u_expect = (bc_right - bc_left) * x / L + bc_left - np.sin(k * x) / k**2
        return max(np.abs(u - u_expect))

    sizes = [int(128 / 2**i) for i in [-2, -1, 0, 1, 2, 3, 4]]
    # sizes = [6]
    errors = np.array([u_num(size) for size in sizes])
    order = np.array(
        [-np.log(errors[i + 1] / errors[i]) / np.log(sizes[i + 1] / sizes[i]) for i in range(len(sizes) - 1)]
    )
    print(order)
    print(errors)
    print(sizes)


if __name__ == '__main__':
    test_Dirichtlet_bcs_sin(4)
