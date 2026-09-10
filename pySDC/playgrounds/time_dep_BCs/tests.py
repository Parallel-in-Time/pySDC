import pytest


@pytest.mark.parametrize('a', [1, 3.14])
@pytest.mark.parametrize('b', [-1, -3.14])
@pytest.mark.parametrize('t', [1, 3.14])
def test_time_dep_heat_eq(a, b, t):
    from pySDC.playgrounds.time_dep_BCs.heat_eq_time_dep_BCs import Heat1DTimeDependentBCs
    import numpy as np

    problem = Heat1DTimeDependentBCs(a=a, b=b, ft=1)
    u0 = problem.u_exact(0)

    u = problem.solve_system(u0, t, u0, t)

    if not problem.spectral_space:
        u = problem.transform(u)

    expect_boundary = np.empty((2, 2))
    expect_boundary = problem.put_time_dep_BCs_in_rhs(expect_boundary, t)

    # we use T_n(1) = 1 and T_n(-1) = (-1)^n, to compute the values at the boundaries from the spectral representation
    # see Wikipedia for more details: https://en.wikipedia.org/wiki/Chebyshev_polynomials#Roots_and_extrema
    right_boundary = u.sum()
    expect_right_boundary = expect_boundary[0, -2]
    assert np.isclose(
        right_boundary, expect_right_boundary
    ), f'Got {right_boundary} at right boundary but expected {expect_right_boundary} at {t=}'

    left_boundary = u[0, ::2].sum() - u[0, 1::2].sum()
    expect_left_boundary = expect_boundary[0, -1]
    assert np.isclose(
        left_boundary, expect_left_boundary
    ), f'Got {left_boundary} at left boundary but expected {expect_left_boundary} at {t=}'
