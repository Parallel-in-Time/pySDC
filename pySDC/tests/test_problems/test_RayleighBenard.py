import pytest


@pytest.mark.mpi4py
@pytest.mark.parametrize('direction', ['x', 'z', 'mixed'])
@pytest.mark.parametrize('nx', [16])
@pytest.mark.parametrize('nz', [8])
@pytest.mark.parametrize('spectral_space', [True, False])
def test_eval_f(nx, nz, direction, spectral_space):
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard

    P = RayleighBenard(nx=nx, nz=nz, Rayleigh=1, spectral_space=spectral_space)
    iu, iv, ip, iT = P.index(['u', 'v', 'p', 'T'])
    X, Z = P.X, P.Z
    cos, sin = np.cos, np.sin

    kappa = P.kappa
    nu = P.nu

    if direction == 'x':
        y = sin(X * np.pi)
        y_x = cos(X * np.pi) * np.pi
        y_xx = -sin(X * np.pi) * np.pi**2
        y_z = 0
        y_zz = 0
    elif direction == 'z':
        y = Z**2
        y_x = 0
        y_xx = 0
        y_z = 2 * Z
        y_zz = 2.0
    elif direction == 'mixed':
        y = sin(X * np.pi) * Z**2
        y_x = cos(X * np.pi) * np.pi * Z**2
        y_xx = -sin(X * np.pi) * np.pi**2 * Z**2
        y_z = sin(X * np.pi) * 2 * Z
        y_zz = sin(X * np.pi) * 2
    else:
        raise NotImplementedError

    assert np.allclose(P.eval_f(P.u_init), 0), 'Non-zero time derivative in static 0 configuration'

    u = P.u_init
    for i in [iu, iv, iT, ip]:
        u[i][:] = y

    if spectral_space:
        u = P.transform(u)

    f = P.eval_f(u)

    f_expect = P.f_init
    f_expect.expl[iT] = -y * (y_x + y_z)
    f_expect.impl[iT] = kappa * (y_xx + y_zz)
    f_expect.expl[iu] = -y * y_z - y * y_x
    f_expect.impl[iu] = -y_x + nu * (y_xx + y_zz)
    f_expect.expl[iv] = -y * (y_z + y_x)
    f_expect.impl[iv] = -y_z + nu * (y_xx + y_zz) + y
    f_expect.impl[ip] = -(y_x + y_z)

    if spectral_space:
        f.impl = P.itransform(f.impl).real
        f.expl = P.itransform(f.expl).real

    for comp in ['u', 'v', 'T', 'p']:
        i = P.spectral.index(comp)
        assert np.allclose(f.impl[i], f_expect.impl[i]), f'Unexpected implicit function evaluation in component {comp}'
        assert np.allclose(f.expl[i], f_expect.expl[i]), f'Unexpected explicit function evaluation in component {comp}'


@pytest.mark.mpi4py
@pytest.mark.parametrize('nx', [16])
@pytest.mark.parametrize('nz', [4])
@pytest.mark.parametrize('direction', ['x', 'z', 'mixed'])
def test_vorticity(nx, nz, direction):
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard

    assert nz > 3
    assert nx > 8

    P = RayleighBenard(nx=nx, nz=nz, spectral_space=False)
    iu, iv = P.index(['u', 'v'])

    u = P.u_init

    if direction == 'x':
        u[iv] = np.sin(P.X * np.pi)
        u[iu] = np.cos(P.X * np.pi)
        expect = np.cos(P.X * np.pi) * np.pi
    elif direction == 'z':
        u[iv] = P.Z**2
        u[iu] = P.Z**3
        expect = 3 * P.Z**2
    elif direction == 'mixed':
        u[iv] = np.sin(P.X * np.pi) * P.Z**2
        u[iu] = np.cos(P.X * np.pi) * P.Z**3
        expect = np.cos(P.X * np.pi) * np.pi * P.Z**2 + np.cos(P.X * np.pi) * 3 * P.Z**2
    else:
        raise NotImplementedError

    assert np.allclose(P.compute_vorticity(u), expect)


@pytest.mark.mpi4py
@pytest.mark.parametrize('v', [0, 3.14])
def test_Nusselt_numbers(v, nx=5, nz=4):
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard

    BCs = {
        'v_top': v,
        'v_bottom': v,
    }

    P = RayleighBenard(nx=nx, nz=nz, BCs=BCs)

    u = P.u_exact(noise_level=0)

    nusselt = P.compute_Nusselt_numbers(u)
    expect = {'V': 1 + v, 't': 1, 'b': +1 + 2 * v, 'b_no_v': 1, 't_no_v': 1}
    for key in nusselt.keys():
        assert np.isclose(nusselt[key], expect[key])


def test_viscous_dissipation(nx=2**5 + 1, nz=2**3 + 1):
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard

    P = RayleighBenard(nx=nx, nz=nz, spectral_space=False)
    iu, iv = P.index(['u', 'v'])
    X, Z = P.X, P.Z

    u = P.u_init
    u[iu] = np.sin(X * np.pi)
    u[iv] = Z**3

    expect = P.u_init
    expect[iu] = u[iu] * (-np.pi) ** 2 * u[iu]
    expect[iv] = Z**3 * 6 * Z

    viscous_dissipation = P.compute_viscous_dissipation(u)
    assert np.isclose(viscous_dissipation, abs(expect))


def test_buoyancy_computation(nx=9, nz=6):
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard

    P = RayleighBenard(nx=nx, nz=nz, spectral_space=False)
    iT, iv = P.index(['T', 'v'])
    Z = P.Z

    u = P.u_init
    u[iT] = Z - 1
    u[iv] = Z**3

    expect = P.u_init
    expect[iv] = u[iv] * P.Rayleigh * u[iT]

    buoyancy_production = P.compute_buoyancy_generation(u)
    assert np.isclose(buoyancy_production, abs(expect))


@pytest.mark.mpi4py
@pytest.mark.parametrize('nx', [1, 8])
@pytest.mark.parametrize('component', ['u', 'T'])
def test_Poisson_problems(nx, component):
    """
    When forgetting about convection and the time-dependent part, you get Poisson problems in u and T that are easy to solve. We check that we get the exact solution in a simple test here.
    """
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard

    BCs = {
        'u_top': 0,
        'u_bottom': 0,
        'v_top': 0,
        'v_bottom': 0,
        'T_top': 0,
        'T_bottom': 0,
    }
    P = RayleighBenard(
        nx=nx, nz=6, BCs=BCs, Rayleigh=(max([abs(BCs['T_top'] - BCs['T_bottom']), np.finfo(float).eps]) * 2**3)
    )
    rhs = P.u_init

    idx = P.index(f'{component}')

    A = P.put_BCs_in_matrix(-P.L)
    rhs[idx][0, 2] = 6
    rhs[idx][0, 0] = 6
    u = P.sparse_lib.linalg.spsolve(A, P.M @ rhs.flatten()).reshape(rhs.shape).real

    u_exact = P.u_init
    u_exact[idx][0, 4] = 1 / 8
    u_exact[idx][0, 2] = 1 / 2
    u_exact[idx][0, 0] = -5 / 8

    if component == 'T':
        ip = P.index('p')
        u_exact[ip][0, 5] = 1 / (16 * 5)
        u_exact[ip][0, 3] = 5 / (16 * 5)
        u_exact[ip][0, 1] = -70 / (16 * 5)

    assert np.allclose(u_exact, u)


@pytest.mark.mpi4py
def test_Poisson_problem_v():
    """
    Here we don't really solve a Poisson problem. v can only be constant due to the incompressibility, then we have a Possion problem in T with a linear solution and p absorbs all the rest. This is therefore mainly a test for the pressure computation. We don't test that the boundary condition is enforced because the constant pressure offset is entirely irrelevant to anything.
    """
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard

    BCs = {
        'u_top': 0,
        'u_bottom': 0,
        'v_top': 0,
        'v_bottom': 0,
        'T_top': 0,
        'T_bottom': 2,
    }
    P = RayleighBenard(nx=2, nz=2**3, BCs=BCs, Rayleigh=1.0)
    iv = P.index('v')

    rhs_real = P.u_init
    rhs_real[iv] = 32 * 6 * P.Z**5

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
    assert np.allclose(u_exact, u)


@pytest.mark.mpi4py
def test_CFL():
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard, CFLLimit
    import numpy as np

    P = RayleighBenard(nx=5, nz=2, spectral_space=False)
    iu, iv = P.index(['u', 'v'])

    u = P.u_init
    u[iu] = 2.77
    u[iv] = 1e-3

    dt = CFLLimit.compute_max_step_size(P, u)
    assert np.allclose(dt, P.X[1, 0] / u[iu])

    u2 = P.u_init
    u2[iu] = 1e-3
    u2[iv] = 3.14

    dt2 = CFLLimit.compute_max_step_size(P, u2)
    assert np.allclose(dt2, 1 / u2[iv])


@pytest.mark.mpi4py
def test_Nyquist_mode_elimination():
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard
    import numpy as np

    P = RayleighBenard(nx=32, nz=8)
    u0 = P.u_exact(noise_level=1e-3)

    u = P.solve_system(u0, dt=1e-3)

    Nyquist_mode_index = P.axes[0].get_Nyquist_mode_index()
    assert np.allclose(u[:, Nyquist_mode_index, :], 0)


@pytest.mark.mpi4py
def test_apply_BCs():
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard
    import numpy as np

    BCs = {
        'u_top': np.random.rand(),
        'u_bottom': np.random.rand(),
        'v_top': np.random.rand(),
        'v_bottom': np.random.rand(),
        'T_top': np.random.rand(),
        'T_bottom': np.random.rand(),
    }
    P = RayleighBenard(nx=5, nz=2**2, BCs=BCs)

    u_in = P.u_init
    u_in[...] = np.random.rand(*u_in.shape)
    u_in_hat = P.transform(u_in)

    u_hat = P.apply_BCs(u_in_hat)
    u = P.itransform(u_hat)

    P.check_BCs(u)


if __name__ == '__main__':
    # test_eval_f(2**0, 2**2, 'z', True)
    # test_Poisson_problem(1, 'T')
    # test_Poisson_problem_v()
    test_apply_BCs()
    # test_Nusselt_numbers(1)
    # test_buoyancy_computation()
    # test_viscous_dissipation()
    # test_CFL()
    # test_Nyquist_mode_elimination()
