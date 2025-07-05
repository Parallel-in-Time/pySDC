import pytest


@pytest.mark.mpi4py
@pytest.mark.parametrize('direction', ['x', 'y', 'z', 'mixed'])
@pytest.mark.parametrize('nx', [16])
@pytest.mark.parametrize('nz', [8])
@pytest.mark.parametrize('spectral_space', [True, False])
def test_eval_f(nx, nz, direction, spectral_space):
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard3D import RayleighBenard3D

    P = RayleighBenard3D(nx=nx, ny=nx, nz=nz, Rayleigh=1, spectral_space=spectral_space)
    iu, iv, iw, ip, iT = P.index(['u', 'v', 'w', 'p', 'T'])
    X, Y, Z = P.X, P.Y, P.Z
    cos, sin = np.cos, np.sin

    kappa = P.kappa
    nu = P.nu

    k = 2
    if direction == 'x':
        y = sin(X * k * np.pi)
        y_x = cos(X * k * np.pi) * k * np.pi
        y_xx = -sin(X * k * np.pi) * k * k * np.pi**2
        y_y = 0
        y_yy = 0
        y_z = 0
        y_zz = 0
    elif direction == 'y':
        y = sin(Y * k * np.pi)
        y_y = cos(Y * k * np.pi) * k * np.pi
        y_yy = -sin(Y * k * np.pi) * k * k * np.pi**2
        y_x = 0
        y_xx = 0
        y_z = 0
        y_zz = 0
    elif direction == 'z':
        y = Z**2
        y_x = 0
        y_xx = 0
        y_y = 0
        y_yy = 0
        y_z = 2 * Z
        y_zz = 2.0
    elif direction == 'mixed':
        y = sin(X * k * np.pi) * sin(Y * k * np.pi) * Z**2
        y_x = cos(X * k * np.pi) * k * np.pi * sin(k * Y * np.pi) * Z**2
        y_xx = -sin(X * k * np.pi) * k * k * np.pi**2 * sin(Y * k * np.pi) * Z**2
        y_y = cos(Y * k * np.pi) * k * np.pi * sin(X * k * np.pi) * Z**2
        y_yy = -sin(Y * k * np.pi) * k * k * np.pi**2 * sin(X * k * np.pi) * Z**2
        y_z = sin(X * k * np.pi) * sin(Y * k * np.pi) * 2 * Z
        y_zz = sin(X * k * np.pi) * sin(Y * k * np.pi) * 2
    else:
        raise NotImplementedError

    u = P.u_init_physical
    assert np.allclose(P.eval_f(P.u_init), 0), 'Non-zero time derivative in static 0 configuration'

    for i in [iu, iv, iw, iT, ip]:
        u[i][:] = y

    if spectral_space:
        u = P.transform(u)

    f = P.eval_f(u)

    f_expect = P.f_init
    for i in [iT, iu, iv, iw]:
        f_expect.expl[i] = -y * (y_x + y_y + y_z)
    f_expect.impl[iT] = kappa * (y_xx + y_yy + y_zz)
    f_expect.impl[iu] = -y_x + nu * (y_xx + y_yy + y_zz)
    f_expect.impl[iv] = -y_y + nu * (y_xx + y_yy + y_zz)
    f_expect.impl[iw] = -y_z + nu * (y_xx + y_yy + y_zz) + y
    f_expect.impl[ip] = -(y_x + y_y + y_z)

    if spectral_space:
        f.impl = P.itransform(f.impl)
        f.expl = P.itransform(f.expl)

    for comp in ['u', 'v', 'w', 'T', 'p'][::-1]:
        i = P.spectral.index(comp)
        assert np.allclose(f.expl[i], f_expect.expl[i]), f'Unexpected explicit function evaluation in component {comp}'
        assert np.allclose(f.impl[i], f_expect.impl[i]), f'Unexpected implicit function evaluation in component {comp}'


@pytest.mark.mpi4py
@pytest.mark.parametrize('direction', ['x', 'y', 'z', 'mixed'])
@pytest.mark.mpi(ranks=[2, 4])
def test_eval_f_parallel(mpi_ranks, direction):
    test_eval_f(nx=4, nz=4, direction=direction, spectral_space=False)


@pytest.mark.mpi4py
@pytest.mark.parametrize('nx', [1, 8])
@pytest.mark.parametrize('component', ['u', 'v', 'T'])
def test_Poisson_problems(nx, component):
    """
    When forgetting about convection and the time-dependent part, you get Poisson problems in u and T that are easy to solve. We check that we get the exact solution in a simple test here.
    """
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard3D import RayleighBenard3D

    BCs = {
        'u_top': 0,
        'u_bottom': 0,
        'v_top': 0,
        'v_bottom': 0,
        'w_top': 0,
        'w_bottom': 0,
        'T_top': 0,
        'T_bottom': 0,
    }
    P = RayleighBenard3D(
        nx=nx, ny=nx, nz=6, BCs=BCs, Rayleigh=(max([abs(BCs['T_top'] - BCs['T_bottom']), np.finfo(float).eps]) * 2**4)
    )
    rhs = P.u_init

    idx = P.index(f'{component}')

    A = P.put_BCs_in_matrix(-P.L)
    rhs[idx][0, 0, 2] = 6
    rhs[idx][0, 0, 0] = 6
    rhs = P.put_BCs_in_rhs_hat(rhs)
    u = P.sparse_lib.linalg.spsolve(A, P.M @ rhs.flatten()).reshape(rhs.shape).real

    u_exact = P.u_init_forward.astype('d')
    if P.comm.rank == 0:
        u_exact[idx][0, 0, 4] = 1 / 8
        u_exact[idx][0, 0, 2] = 1 / 2
        u_exact[idx][0, 0, 0] = -5 / 8

        if component == 'T':
            ip = P.index('p')
            u_exact[ip][0, 0, 5] = 1 / (16 * 5) / 2
            u_exact[ip][0, 0, 3] = 5 / (16 * 5) / 2
            u_exact[ip][0, 0, 1] = -70 / (16 * 5) / 2

    for comp in ['u', 'v', 'w', 'T', 'p']:
        i = P.spectral.index(comp)
        assert np.allclose(u_exact[i], u[i]), f'Unexpected solution in component {comp}'


@pytest.mark.mpi4py
def test_Poisson_problem_w():
    """
    Here we don't really solve a Poisson problem. w can only be constant due to the incompressibility, then we have a Possion problem in T with a linear solution and p absorbs all the rest. This is therefore mainly a test for the pressure computation. We don't test that the boundary condition is enforced because the constant pressure offset is entirely irrelevant to anything.
    """
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard3D import RayleighBenard3D

    BCs = {
        'u_top': 0,
        'u_bottom': 0,
        'v_top': 0,
        'v_bottom': 0,
        'w_top': 0,
        'w_bottom': 0,
        'T_top': 0,
        'T_bottom': 1,
    }
    P = RayleighBenard3D(nx=2, ny=2, nz=2**3, BCs=BCs, Rayleigh=1.0)
    iw = P.index('w')

    rhs_real = P.u_init
    rhs_real[iw] = 32 * 6 * P.Z**5

    rhs = P.transform(rhs_real)
    rhs = (P.M @ rhs.flatten()).reshape(rhs.shape)
    rhs_real = P.itransform(rhs)

    rhs_real = P.put_BCs_in_rhs(rhs_real)
    rhs = P.transform(rhs_real)

    A = P.put_BCs_in_matrix(-P.L)
    u = P.sparse_lib.linalg.spsolve(A, rhs.flatten()).reshape(rhs.shape).real

    u_exact_real = P.u_init
    iT = P.index('T')
    u_exact_real[iT] = 1 - P.Z

    ip = P.index('p')
    u_exact_real[ip] = P.Z - 1 / 2 * P.Z**2 - 32 * P.Z**6

    u_exact = P.transform(u_exact_real)
    u_exact[ip, 0, 0] = u[ip, 0, 0]  # nobody cares about the constant offset

    for comp in ['u', 'v', 'w', 'T', 'p']:
        i = P.spectral.index(comp)
        assert np.allclose(u_exact[i], u[i]), f'Unexpected solution in component {comp}'


@pytest.mark.mpi4py
def test_libraries():
    from pySDC.implementations.problem_classes.RayleighBenard3D import RayleighBenard3D
    from mpi4py_fft import fftw
    from scipy import fft

    P = RayleighBenard3D(nx=2, ny=2, nz=2)
    assert P.axes[0].fft_lib == fftw
    assert P.axes[1].fft_lib == fftw
    assert P.axes[2].fft_lib == fft


if __name__ == '__main__':
    # test_eval_f(2**2, 2**1, 'x', False)
    # test_libraries()
    # test_Poisson_problems(4, 'u')
    test_Poisson_problem_w()
