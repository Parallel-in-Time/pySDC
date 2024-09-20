import pytest


@pytest.mark.base
@pytest.mark.parametrize('N', [9, 64])
@pytest.mark.parametrize('x0', [-4, 0, 1])
@pytest.mark.parametrize('x1', [None, 4, 8])
def test_differentiation_matrix(N, x0, x1, plot=False):
    import numpy as np
    from pySDC.helpers.spectral_helper import FFTHelper

    x1 = 2 * np.pi if x1 is None else x1
    helper = FFTHelper(N=N, x0=x0, x1=x1)

    x = helper.get_1dgrid()
    D = helper.get_differentiation_matrix()

    u = np.zeros_like(x)
    expect = np.zeros_like(u)

    num_coef = N // 2
    f = 2 * np.pi / helper.L
    coeffs = np.random.random((2, N))
    for i in range(num_coef):
        u += coeffs[0, i] * np.sin(i * x * f)
        u += coeffs[1, i] * np.cos(i * x * f)
        expect += coeffs[0, i] * i * np.cos(i * x * f) * f
        expect -= coeffs[1, i] * i * np.sin(i * x * f) * f

    u_hat = np.fft.fft(u)
    Du_hat = D @ u_hat
    Du = np.fft.ifft(Du_hat)

    if plot:
        import matplotlib.pyplot as plt

        plt.plot(x, u)
        plt.plot(x, Du)
        plt.plot(x, expect, '--')
        plt.show()

    assert np.allclose(expect, Du)


@pytest.mark.base
def test_transform(N=8):
    import numpy as np
    from pySDC.helpers.spectral_helper import FFTHelper

    u = np.random.random(N)
    helper = FFTHelper(N=N)
    u_hat = helper.transform(u)
    assert np.allclose(u, helper.itransform(u_hat))


@pytest.mark.base
@pytest.mark.parametrize('N', [8, 64])
def test_integration_matrix(N, plot=False):
    import numpy as np
    from pySDC.helpers.spectral_helper import FFTHelper

    helper = FFTHelper(N=N)

    x = helper.get_1dgrid()
    D = helper.get_integration_matrix()

    u = np.zeros_like(x)
    expect = np.zeros_like(u)

    num_coef = N // 2 - 1
    coeffs = np.random.random((2, N))
    for i in range(1, num_coef + 1):
        u += coeffs[0, i] * np.sin(i * x)
        u += coeffs[1, i] * np.cos(i * x)
        expect -= coeffs[0, i] / i * np.cos(i * x)
        expect += coeffs[1, i] / i * np.sin(i * x)

    u_hat = np.fft.fft(u)
    Du_hat = D @ u_hat
    Du = np.fft.ifft(Du_hat)

    if plot:
        import matplotlib.pyplot as plt

        plt.plot(x, u)
        plt.plot(x, Du)
        plt.plot(x, expect, '--')
        plt.show()

    assert np.allclose(expect, Du)


@pytest.mark.base
@pytest.mark.parametrize('N', [4, 32])
@pytest.mark.parametrize('v', [0, 4.78])
def test_tau_method(N, v):
    import numpy as np
    from pySDC.helpers.spectral_helper import FFTHelper

    helper = FFTHelper(N=N)

    D = helper.get_differentiation_matrix()
    bc_line = 0
    BC = (helper.get_Id() * 0).tolil()
    BC[bc_line, :] = helper.get_integ_BC_row()

    A = D + BC
    rhs = np.zeros(N)
    rhs[bc_line] = v
    u_hat = helper.sparse_lib.linalg.spsolve(A, rhs)

    u = helper.itransform(u_hat)
    assert np.allclose(u * helper.L, v)


if __name__ == '__main__':
    # test_differentiation_matrix(64, 4, True)
    # test_integration_matrix(8, True)
    test_tau_method(6, 1)
