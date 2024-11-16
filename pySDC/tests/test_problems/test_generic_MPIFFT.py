import pytest


@pytest.mark.mpi4py
@pytest.mark.parametrize('nx', [8, 16])
@pytest.mark.parametrize('ny', [8, 16])
@pytest.mark.parametrize('nz', [0, 8])
@pytest.mark.parametrize('f', [1, 3])
@pytest.mark.parametrize('spectral', [True, False])
@pytest.mark.parametrize('direction', [0, 1, 10])
def test_derivative(nx, ny, nz, f, spectral, direction):
    from pySDC.implementations.problem_classes.generic_MPIFFT_Laplacian import IMEX_Laplacian_MPIFFT

    nvars = (nx, ny)
    if nz > 0:
        nvars += (nz,)
    prob = IMEX_Laplacian_MPIFFT(nvars=nvars, spectral=spectral)

    xp = prob.xp

    if direction == 0:
        _u = xp.sin(f * prob.X[0])
        du_expect = -(f**2) * xp.sin(f * prob.X[0])
    elif direction == 1:
        _u = xp.sin(f * prob.X[1])
        du_expect = -(f**2) * xp.sin(f * prob.X[1])
    elif direction == 10:
        _u = xp.sin(f * prob.X[1]) + xp.cos(f * prob.X[0])
        du_expect = -(f**2) * xp.sin(f * prob.X[1]) - f**2 * xp.cos(f * prob.X[0])
    else:
        raise

    if spectral:
        u = prob.fft.forward(_u)
    else:
        u = _u

    _du = prob.eval_f(u, 0).impl

    if spectral:
        du = prob.fft.backward(_du)
    else:
        du = _du
    assert xp.allclose(du, du_expect), 'Got unexpected derivative'

    u2 = prob.solve_system(_du, factor=1e8, u0=du, t=0) * -1e8

    if spectral:
        _u2 = prob.fft.backward(u2)
    else:
        _u2 = u2

    assert xp.allclose(_u2, _u, atol=1e-7), 'Got unexpected inverse derivative'


if __name__ == '__main__':
    test_derivative(6, 6, 6, 3, False, 1)
