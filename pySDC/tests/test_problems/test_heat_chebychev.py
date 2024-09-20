import pytest


@pytest.mark.base
@pytest.mark.parametrize('a', [0, 19])
@pytest.mark.parametrize('b', [0, 66])
@pytest.mark.parametrize('f', [0, 1])
@pytest.mark.parametrize('noise', [0, 1e-3])
@pytest.mark.parametrize('use_ultraspherical', [True, False])
@pytest.mark.parametrize('spectral_space', [True, False])
def test_heat1d_chebychev(a, b, f, noise, use_ultraspherical, spectral_space, nvars=2**4):
    import numpy as np

    if use_ultraspherical:
        from pySDC.implementations.problem_classes.HeatEquation_Chebychev import Heat1DUltraspherical as problem_class
    else:
        from pySDC.implementations.problem_classes.HeatEquation_Chebychev import Heat1DChebychev as problem_class

    P = problem_class(
        nvars=nvars,
        a=a,
        b=b,
        f=f,
        nu=1e-2,
        left_preconditioner=False,
        debug=True,
        spectral_space=spectral_space,
    )

    u0 = P.u_exact(0, noise=noise)
    dt = 1e-1
    u = P.solve_system(u0, dt)
    u02 = u - dt * P.eval_f(u)

    if noise == 0:
        assert np.allclose(u, P.u_exact(dt), atol=1e-2), 'Error in solver'

    if noise > 0 and use_ultraspherical:
        tol = 2e-2
    elif noise > 0:
        tol = 1e-4
    else:
        tol = 1e-8
    assert np.allclose(u0[0], u02[0], atol=tol), 'Error in eval_f'


@pytest.mark.base
@pytest.mark.parametrize('a', [0, 7])
@pytest.mark.parametrize('b', [0, -2.77])
@pytest.mark.parametrize('c', [0, 3.1415])
@pytest.mark.parametrize('fx', [2, 1])
@pytest.mark.parametrize('fy', [2, 1])
@pytest.mark.parametrize('base_x', ['fft', 'chebychev', 'ultraspherical'])
@pytest.mark.parametrize('base_y', ['fft', 'chebychev', 'ultraspherical'])
def test_heat2d_chebychev(a, b, c, fx, fy, base_x, base_y, nx=2**5 + 1, ny=2**5 + 1):
    import numpy as np

    if base_x == 'ultraspherical' or base_y == 'ultraspherical':
        from pySDC.implementations.problem_classes.HeatEquation_Chebychev import Heat2DUltraspherical as problem_class
    else:
        from pySDC.implementations.problem_classes.HeatEquation_Chebychev import Heat2DChebychev as problem_class

    if base_x == 'chebychev' and base_y == 'ultraspherical' or base_y == 'chebychev' and base_x == 'ultraspherical':
        return None

    if base_y == 'fft' and (b != c):
        return None
    if base_x == 'fft' and (b != a):
        return None

    P = problem_class(
        nx=nx,
        ny=ny,
        a=a,
        b=b,
        c=c,
        fx=fx,
        fy=fy,
        base_x=base_x,
        base_y=base_y,
        nu=1e-3,
    )

    u0 = P.u_exact(0)
    dt = 1e-2
    u = P.solve_system(u0, dt)
    u02 = u - dt * P.eval_f(u)

    assert np.allclose(
        u, P.u_exact(dt), atol=1e-3
    ), f'Error in solver larger than expected, got {abs((u - P.u_exact(dt))):.2e}'
    assert np.allclose(u0[0], u02[0], atol=1e-4), 'Error in eval_f'


def test_SDC():
    import numpy as np
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.problem_classes.HeatEquation_Chebychev import Heat1DChebychev
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.problem_classes.generic_spectral import compute_residual_DAE
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.hooks.log_work import LogSDCIterations

    generic_implicit.compute_residual = compute_residual_DAE

    dt = 1e-1
    Tend = 2 * dt

    level_params = {}
    level_params['dt'] = dt
    level_params['restol'] = 1e-10

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 4
    sweeper_params['QI'] = 'LU'

    problem_params = {}

    step_params = {}
    step_params['maxiter'] = 99

    controller_params = {}
    controller_params['logger_level'] = 15
    controller_params['hook_class'] = [LogSDCIterations]

    description = {}
    description['problem_class'] = Heat1DChebychev
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t=0)

    uend, stats = controller.run(u0=uinit, t0=0.0, Tend=Tend)
    u_exact = P.u_exact(t=Tend)
    assert np.allclose(uend, u_exact, atol=1e-10)

    k = get_sorted(stats, type='k')
    assert all(me[1] < step_params['maxiter'] for me in k)
    assert all(me[1] > 0 for me in k)


if __name__ == '__main__':
    test_SDC()
    # test_heat1d_chebychev(1, 0, 1, 0e-3, True, True, 2**4)
    # test_heat2d_chebychev(0, 0, 0, 2, 2, 'ultraspherical', 'fft', 2**6, 2**6)
