import pytest


@pytest.mark.base
@pytest.mark.parametrize('N', [4, 7, 32])
@pytest.mark.parametrize('p', [1, 2, 3, 4])
def test_differentiation_matrix(N, p):
    import numpy as np
    from pySDC.helpers.spectral_helper import UltrasphericalHelper

    helper = UltrasphericalHelper(N)
    x = helper.get_1dgrid()
    coeffs = np.random.random(N)

    D = helper.get_differentiation_matrix(p=p)
    Q = helper.get_basis_change_matrix(p_in=p, p_out=0)

    du = helper.itransform(Q @ D @ coeffs, axes=(-1,))
    exact = np.polynomial.Chebyshev(coeffs).deriv(p)(x)

    assert np.allclose(exact, du)


@pytest.mark.base
@pytest.mark.parametrize('N', [4, 7, 32])
@pytest.mark.parametrize('x0', [-1, 0])
@pytest.mark.parametrize('x1', [0.789, 1])
@pytest.mark.parametrize('p', [1, 2])
def test_differentiation_non_standard_domain_size(N, x0, x1, p):
    import numpy as np
    import scipy
    from pySDC.helpers.spectral_helper import UltrasphericalHelper

    helper = UltrasphericalHelper(N, x0=x0, x1=x1)
    x = helper.get_1dgrid()
    assert all(x > x0)
    assert all(x < x1)

    coeffs = np.random.random(N)
    u = np.polynomial.Chebyshev(coeffs)(x)
    u_hat = helper.transform(u)
    du_exact = np.polynomial.Chebyshev(coeffs).deriv(p)(x)
    du_hat_exact = helper.transform(du_exact)

    D = helper.get_differentiation_matrix(p)
    Q = helper.get_basis_change_matrix(p_in=p, p_out=0)

    du_hat = Q @ D @ u_hat
    du = helper.itransform(du_hat, axes=(-1,))

    assert np.allclose(du_hat_exact, du_hat), np.linalg.norm(du_hat_exact - du_hat)
    assert np.allclose(du, du_exact), np.linalg.norm(du_exact - du)


@pytest.mark.base
@pytest.mark.parametrize('N', [4, 7, 32])
def test_integration(N):
    import numpy as np
    from pySDC.helpers.spectral_helper import UltrasphericalHelper

    helper = UltrasphericalHelper(N)
    coeffs = np.random.random(N)
    coeffs[-1] = 0

    poly = np.polynomial.Chebyshev(coeffs)

    S = helper.get_integration_matrix()
    U_hat = S @ coeffs
    U_hat[0] = helper.get_integration_constant(U_hat, axis=-1)

    assert np.allclose(poly.integ(lbnd=-1).coef[:-1], U_hat)


@pytest.mark.base
@pytest.mark.parametrize('N', [4, 7, 32])
@pytest.mark.parametrize('x0', [-1, 0])
def test_integration_rescaled_domain(N, x0, x1=1):
    import numpy as np
    from pySDC.helpers.spectral_helper import UltrasphericalHelper

    helper = UltrasphericalHelper(N, x0=x0, x1=x1)

    x = helper.get_1dgrid()
    y = N - 2
    u = x**y * (y + 1)
    u_hat = helper.transform(u)

    S = helper.get_integration_matrix()
    U_hat = S @ u_hat
    U_hat[0] = helper.get_integration_constant(U_hat, axis=-1)

    int_expect = x ** (y + 1)
    int_hat_expect = helper.transform(int_expect)

    # compare indefinite integral
    assert np.allclose(int_hat_expect[1:], U_hat[1:])
    if x0 == 0:
        assert np.allclose(int_hat_expect[0], U_hat[0]), 'Integration constant is wrong!'

    # compare definite integral
    top = helper.get_BC(kind='dirichlet', x=1)
    bot = helper.get_BC(kind='dirichlet', x=-1)
    res = ((top - bot) * U_hat).sum()
    if x0 == 0:
        assert np.isclose(res, 1)
    elif x0 == -1:
        assert np.isclose(res, 0 if y % 2 == 1 else 2)


@pytest.mark.base
@pytest.mark.parametrize('N', [6, 33])
@pytest.mark.parametrize('deg', [1, 3])
@pytest.mark.parametrize('Dirichlet_recombination', [False, True])
def test_poisson_problem(N, deg, Dirichlet_recombination):
    import numpy as np
    import scipy.sparse as sp
    from pySDC.helpers.spectral_helper import UltrasphericalHelper

    a = 0
    b = 4

    helper = UltrasphericalHelper(N)
    x = helper.get_1dgrid()

    f = x**deg * (deg + 1) * (deg + 2) * (a - b) / 2

    Dxx = helper.get_differentiation_matrix(p=2)
    BC_l = helper.get_Dirichlet_BC_row(x=-1)
    BC_r = helper.get_Dirichlet_BC_row(x=1)
    P = helper.get_basis_change_matrix(p_in=0, p_out=2)

    A = Dxx.tolil()
    A[-1, :] = BC_l
    A[-2, :] = BC_r
    A = A.tocsr()

    if Dirichlet_recombination:
        Pr = helper.get_Dirichlet_recombination_matrix()

        BC_D_r = np.zeros(N)
        BC_D_r[0] = 1
        BC_D_r[1] = 1

        BC_D_l = np.zeros(N)
        BC_D_l[0] = 1
        BC_D_l[1] = -1
        assert np.allclose((A @ Pr).toarray()[-1], BC_D_l)
        assert np.allclose((A @ Pr).toarray()[-2], BC_D_r)
    else:
        Pr = helper.get_Id()

    rhs = P @ helper.transform(f)
    rhs[-2] = a
    rhs[-1] = b

    u_hat = Pr @ sp.linalg.spsolve(A @ Pr, rhs)

    u = helper.itransform(u_hat, axes=(-1,))

    u_exact = (a - b) / 2 * x ** (deg + 2) + (b + a) / 2

    assert np.allclose(u_hat[deg + 3 :], 0)
    assert np.allclose(u_exact, u)
