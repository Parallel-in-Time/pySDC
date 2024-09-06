import pytest


@pytest.mark.base
@pytest.mark.parametrize('N', [4, 32])
def test_D2T_conversion_matrices(N):
    import numpy as np
    from pySDC.helpers.spectral_helper import ChebychevHelper

    cheby = ChebychevHelper(N)

    x = np.linspace(-1, 1, N)
    D2T = cheby.get_conv('D2T')

    for i in range(N):
        coeffs = np.zeros(N)
        coeffs[i] = 1.0
        T_coeffs = D2T @ coeffs

        Dn = np.polynomial.Chebyshev(T_coeffs)(x)

        expect_left = (-1) ** i if i < 2 else 0
        expect_right = 1 if i < 2 else 0

        assert Dn[0] == expect_left
        assert Dn[-1] == expect_right


@pytest.mark.base
@pytest.mark.parametrize('N', [4, 32])
def test_T_U_conversion(N):
    import numpy as np
    from scipy.special import chebyt, chebyu
    from pySDC.helpers.spectral_helper import ChebychevHelper

    cheby = ChebychevHelper(N)

    T2U = cheby.get_conv('T2U')
    U2T = cheby.get_conv('U2T')

    coeffs = np.random.random(N)
    x = cheby.get_1dgrid()

    def eval_poly(poly, coeffs, x):
        return np.array([coeffs[i] * poly(i)(x) for i in range(len(coeffs))]).sum(axis=0)

    u = eval_poly(chebyu, coeffs, x)
    t_from_u = eval_poly(chebyt, U2T @ coeffs, x)
    t_from_u_r = eval_poly(chebyt, coeffs @ U2T.T, x)

    t = eval_poly(chebyt, coeffs, x)
    u_from_t = eval_poly(chebyu, T2U @ coeffs, x)
    u_from_t_r = eval_poly(chebyu, coeffs @ T2U.T, x)

    assert np.allclose(u, t_from_u)
    assert np.allclose(u, t_from_u_r)
    assert np.allclose(t, u_from_t)
    assert np.allclose(t, u_from_t_r)


@pytest.mark.base
@pytest.mark.parametrize('name', ['T2U', 'T2D', 'T2T'])
def test_conversion_inverses(name):
    from pySDC.helpers.spectral_helper import ChebychevHelper
    import numpy as np

    N = 8
    cheby = ChebychevHelper(N)
    P = cheby.get_conv(name)
    Pinv = cheby.get_conv(name[::-1])
    assert np.allclose((P @ Pinv).toarray(), np.diag(np.ones(N)))


@pytest.mark.base
@pytest.mark.parametrize('N', [4, 32])
def test_differentiation_matrix(N):
    import numpy as np
    import scipy
    from pySDC.helpers.spectral_helper import ChebychevHelper

    cheby = ChebychevHelper(N)
    x = np.cos(np.pi / N * (np.arange(N) + 0.5))
    coeffs = np.random.random(N)
    norm = cheby.get_norm()

    D = cheby.get_differentiation_matrix(1)

    du = scipy.fft.idct(D @ coeffs / norm)
    exact = np.polynomial.Chebyshev(coeffs).deriv(1)(x)

    assert np.allclose(exact, du)


@pytest.mark.base
@pytest.mark.parametrize('N', [4, 32])
def test_integration_matrix(N):
    import numpy as np
    from pySDC.helpers.spectral_helper import ChebychevHelper

    cheby = ChebychevHelper(N)
    coeffs = np.random.random(N)
    coeffs[-1] = 0

    D = cheby.get_integration_matrix()

    du = D @ coeffs
    exact = np.polynomial.Chebyshev(coeffs).integ(1, lbnd=0)

    assert np.allclose(exact.coef[:-1], du)


@pytest.mark.base
@pytest.mark.parametrize('N', [4])
@pytest.mark.parametrize('d', [1, 2, 3])
@pytest.mark.parametrize('transform_type', ['dct', 'fft'])
def test_transform(N, d, transform_type):
    import scipy
    import numpy as np
    from pySDC.helpers.spectral_helper import ChebychevHelper

    cheby = ChebychevHelper(N, transform_type=transform_type)
    u = np.random.random((d, N))
    norm = cheby.get_norm()
    x = cheby.get_1dgrid()

    itransform = cheby.itransform(u, axis=-1).real

    for i in range(d):
        assert np.allclose(np.polynomial.Chebyshev(u[i])(x), itransform[i])
    assert np.allclose(u.shape, itransform.shape)
    assert np.allclose(scipy.fft.dct(u, axis=-1) * norm, cheby.transform(u, axis=-1).real)
    assert np.allclose(scipy.fft.idct(u / norm, axis=-1), itransform)
    assert np.allclose(cheby.transform(cheby.itransform(u)), u)
    assert np.allclose(cheby.itransform(cheby.transform(u)), u)


@pytest.mark.base
@pytest.mark.parametrize('N', [8, 32])
def test_integration_BC(N):
    from pySDC.helpers.spectral_helper import ChebychevHelper
    import numpy as np

    cheby = ChebychevHelper(N)
    coef = np.random.random(N)

    BC = cheby.get_integ_BC_row()

    polynomial = np.polynomial.Chebyshev(coef)
    reference_integral = polynomial.integ(lbnd=-1, k=0)

    integral = sum(coef * BC)
    assert np.isclose(integral, reference_integral(1))


@pytest.mark.base
@pytest.mark.parametrize('N', [4, 32])
def test_norm(N):
    from pySDC.helpers.spectral_helper import ChebychevHelper
    import numpy as np
    import scipy

    cheby = ChebychevHelper(N)
    coeffs = np.random.random(N)
    x = cheby.get_1dgrid()
    norm = cheby.get_norm()

    u = np.polynomial.Chebyshev(coeffs)(x)
    u_dct = scipy.fft.idct(coeffs / norm)
    coeffs_dct = scipy.fft.dct(u) * norm

    assert np.allclose(u, u_dct)
    assert np.allclose(coeffs, coeffs_dct)


@pytest.mark.base
@pytest.mark.parametrize('bc', [-1, 0, 1])
@pytest.mark.parametrize('N', [3, 32])
@pytest.mark.parametrize('bc_val', [-99, 3.1415])
def test_tau_method(bc, N, bc_val):
    '''
    solve u_x - u + tau P = 0, u(bc) = bc_val

    We choose P = T_N or U_N. We replace the last row in the matrix with the boundary condition to get a
    unique solution for the given resolution.

    The test verifies that the solution satisfies the perturbed equation and the boundary condition.
    '''
    from pySDC.helpers.spectral_helper import ChebychevHelper
    import numpy as np

    cheby = ChebychevHelper(N)
    x = cheby.get_1dgrid()

    coef = np.append(np.zeros(N - 1), [1])
    rhs = np.append(np.zeros(N - 1), [bc_val])

    P = np.polynomial.Chebyshev(coef)
    D = cheby.get_differentiation_matrix()
    Id = np.diag(np.ones(N))

    A = D - Id
    A[-1, :] = cheby.get_Dirichlet_BC_row(bc)

    sol_hat = np.linalg.solve(A, rhs)

    sol_poly = np.polynomial.Chebyshev(sol_hat)
    d_sol_poly = sol_poly.deriv(1)
    x = np.linspace(-1, 1, 100)

    assert np.isclose(sol_poly(bc), bc_val), 'Solution does not satisfy boundary condition'

    tau = (d_sol_poly(x) - sol_poly(x)) / P(x)
    assert np.allclose(tau, tau[0]), 'Solution does not satisfy perturbed equation'


@pytest.mark.base
@pytest.mark.parametrize('bc', [-1, 1])
@pytest.mark.parametrize('nx', [4, 8])
@pytest.mark.parametrize('nz', [4, 8])
@pytest.mark.parametrize('bc_val', [-2, 1.0])
def test_tau_method2D(bc, nz, nx, bc_val, plotting=False):
    '''
    solve u_z - 0.1u_xx -u_x + tau P = 0, u(bc) = sin(bc_val*x) -> space-time discretization of advection-diffusion
    problem. We do FFT in x-direction and Chebychev in z-direction.
    '''
    from pySDC.helpers.spectral_helper import ChebychevHelper, FFTHelper
    import numpy as np
    import scipy.sparse as sp

    cheby = ChebychevHelper(nz)
    fft = FFTHelper(nx)

    # generate grid
    x = fft.get_1dgrid()
    z = cheby.get_1dgrid()
    Z, X = np.meshgrid(z, x)

    # put BCs in right hand side
    bcs = np.sin(bc_val * x)
    rhs = np.zeros_like(X)
    rhs[:, -1] = bcs
    rhs_hat = fft.transform(rhs, axis=-2)  # the rhs is already in Chebychev spectral space

    # generate matrices
    Dx = fft.get_differentiation_matrix(p=2) * 1e-1 + fft.get_differentiation_matrix()
    Ix = fft.get_Id()
    Dz = cheby.get_differentiation_matrix()
    Iz = cheby.get_Id()
    A = sp.kron(Ix, Dz) - sp.kron(Dx, Iz)

    # put BCs in the system matrix
    BCz = sp.eye(nz, format='lil') * 0
    BCz[-1, :] = cheby.get_Dirichlet_BC_row(bc)
    BC = sp.kron(Ix, BCz, format='lil')
    A[BC != 0] = BC[BC != 0]

    # solve the system
    sol_hat = (sp.linalg.spsolve(A, rhs_hat.flatten())).reshape(rhs.shape)

    # transform back to real space
    _sol = fft.itransform(sol_hat, axis=-2).real
    sol = cheby.itransform(_sol, axis=-1)

    # construct polynomials for testing
    polys = [np.polynomial.Chebyshev(_sol[i, :]) for i in range(nx)]
    # d_polys = [me.deriv(1) for me in polys]
    # _z = np.linspace(-1, 1, 100)

    if plotting:
        import matplotlib.pyplot as plt

        im = plt.pcolormesh(X, Z, sol)
        plt.colorbar(im)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.show()

    for i in range(nx):

        assert np.isclose(polys[i](bc), bcs[i]), f'Solution does not satisfy boundary condition x={x[i]}'

        assert np.allclose(
            polys[i](z), sol[i, :]
        ), f'Solution is incorrectly transformed back to real space at x={x[i]}'

        # coef = np.append(np.zeros(nz - 1), [1])
        # Pz = np.polynomial.Chebyshev(coef)
        # tau = (d_polys[i](_z) - polys[i](_z)) / Pz(_z)
        # plt.plot(_z, tau)
        # plt.show()
        # assert np.allclose(tau, tau[0]), f'Solution does not satisfy perturbed equation at x={x[i]}'


@pytest.mark.base
@pytest.mark.parametrize('nx', [4, 8])
@pytest.mark.parametrize('nz', [16])
@pytest.mark.parametrize('bc_val', [4.0])
def test_tau_method2D_diffusion(nz, nx, bc_val, plotting=False):
    '''
    Solve a Poisson problem with funny Dirichlet BCs in z-direction and periodic in x-direction.
    '''
    from pySDC.helpers.spectral_helper import ChebychevHelper, FFTHelper
    import numpy as np
    import scipy.sparse as sp

    cheby = ChebychevHelper(nz)
    fft = FFTHelper(nx)

    # generate grid
    x = fft.get_1dgrid()
    z = cheby.get_1dgrid()
    Z, X = np.meshgrid(z, x)

    # put BCs in right hand side
    rhs = np.zeros((2, nx, nz))  # components u and u_x
    rhs[0, :, -1] = np.sin(bc_val * x) + 1
    rhs[1, :, -1] = 3 * np.exp(-((x - 3.6) ** 2)) + np.cos(x)
    rhs_hat = fft.transform(rhs, axis=-2)  # the rhs is already in Chebychev spectral space

    # generate 1D matrices
    Dx = fft.get_differentiation_matrix()
    Ix = fft.get_Id()
    Dz = cheby.get_differentiation_matrix()
    Iz = cheby.get_Id()

    # generate 2D matrices
    D = sp.kron(Ix, Dz) + sp.kron(Dx, Iz)
    I = sp.kron(Ix, Iz)
    O = I * 0

    # generate system matrix
    A = sp.bmat([[O, D], [D, -I]], format='lil')

    # generate BC matrices
    BCa = sp.eye(nz, format='lil') * 0
    BCa[-1, :] = cheby.get_Dirichlet_BC_row(-1)
    BCa = sp.kron(Ix, BCa, format='lil')

    BCb = sp.eye(nz, format='lil') * 0
    BCb[-1, :] = cheby.get_Dirichlet_BC_row(1)
    BCb = sp.kron(Ix, BCb, format='lil')
    BC = sp.bmat([[BCa, O], [BCb, O]], format='lil')

    # put BCs in the system matrix
    A[BC != 0] = BC[BC != 0]

    # solve the system
    sol_hat = (sp.linalg.spsolve(A, rhs_hat.flatten())).reshape(rhs.shape)

    # transform back to real space
    _sol = fft.itransform(sol_hat, axis=-2).real
    sol = cheby.itransform(_sol, axis=-1)

    polys = [np.polynomial.Chebyshev(_sol[0, i, :]) for i in range(nx)]

    if plotting:
        import matplotlib.pyplot as plt

        im = plt.pcolormesh(X, Z, sol[0])
        plt.colorbar(im)
        plt.xlabel('x')
        plt.ylabel('z')
        plt.show()

    for i in range(nx):

        assert np.isclose(polys[i](-1), rhs[0, i, -1]), f'Solution does not satisfy lower boundary condition x={x[i]}'
        assert np.isclose(polys[i](1), rhs[1, i, -1]), f'Solution does not satisfy upper boundary condition x={x[i]}'

        assert np.allclose(
            polys[i](z), sol[0, i, :]
        ), f'Solution is incorrectly transformed back to real space at x={x[i]}'


if __name__ == '__main__':
    test_differentiation_matrix(4, 'T2U')
    # test_transform(6, 1, 'fft')
    # test_tau_method('T2U', -1.0, N=4, bc_val=3.0)
    # test_tau_method2D('T2T', -1, nx=2**7, nz=2**6, bc_val=4.0, plotting=True)
    # test_integration_matrix(5, 'T2U')
    # test_integration_matrix2D(2**0, 2**2, 'T2U', 'z')
    # test_differentiation_matrix2D(2**7, 2**7, 'T2U', 'mixed')
    # test_integration_BC(6)
    # test_filter(12, 2, 5, 'T2U')
