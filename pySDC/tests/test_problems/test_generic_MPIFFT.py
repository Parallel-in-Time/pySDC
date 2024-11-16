import pytest


@pytest.mark.mpi4py
@pytest.mark.parametrize('nx', [16, 32])
@pytest.mark.parametrize('ny', [16, 32])
@pytest.mark.parametrize('nz', [0])
@pytest.mark.parametrize('f', [1, 3])
@pytest.mark.parametrize('direction', [0, 1, 10])
def test_derivative(nx, ny, nz, f, direction):
    from pySDC.implementations.problem_classes.generic_MPIFFT_Laplacian import IMEX_Laplacian_MPIFFT

    nvars = (nx, ny)
    if nz > 0:
        nvars.append(nz)
    prob = IMEX_Laplacian_MPIFFT(nvars=nvars)

    xp = prob.xp

    if direction == 0:
        u = xp.sin(f * prob.X[0])
        du_expect = -(f**2) * xp.sin(f * prob.X[0])
    elif direction == 1:
        u = xp.sin(f * prob.X[1])
        du_expect = -(f**2) * xp.sin(f * prob.X[1])
    elif direction == 10:
        u = xp.sin(f * prob.X[1]) + xp.cos(f * prob.X[0])
        du_expect = -(f**2) * xp.sin(f * prob.X[1]) - f**2 * xp.cos(f * prob.X[0])
    else:
        raise

    du = prob.eval_f(u, 0).impl
    assert xp.allclose(du, du_expect), 'Got unexpected derivative'

    _u = prob.solve_system(du, factor=1e8, u0=du, t=0) * -1e8
    assert xp.allclose(_u, u, atol=1e-7), 'Got unexpected inverse derivative'


if __name__ == '__main__':
    test_derivative(32, 32, 0, 1, 1)
