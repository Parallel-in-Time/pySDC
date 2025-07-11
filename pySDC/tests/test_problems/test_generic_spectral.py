import pytest


@pytest.mark.base
@pytest.mark.parametrize('use_ultraspherical', [True, False])
@pytest.mark.parametrize('spectral_space', [True, False])
@pytest.mark.parametrize('solver_type', ['bicgstab'])
@pytest.mark.parametrize('left_preconditioner', [True, False])
@pytest.mark.parametrize('Dirichlet_recombination', [True, False])
def test_initial_guess(
    use_ultraspherical, spectral_space, solver_type, left_preconditioner, Dirichlet_recombination, nvars=2**4
):
    import numpy as np

    if use_ultraspherical:
        from pySDC.implementations.problem_classes.HeatEquation_Chebychev import Heat1DUltraspherical as problem_class
    else:
        from pySDC.implementations.problem_classes.HeatEquation_Chebychev import Heat1DChebychev as problem_class

    params = {
        'nvars': nvars,
        'a': 0,
        'b': 1,
        'f': 0,
        'nu': 1e-2,
        'debug': True,
        'spectral_space': spectral_space,
        'solver_type': solver_type,
        'solver_args': {'rtol': 1e-12},
        'left_preconditioner': left_preconditioner,
        'Dirichlet_recombination': Dirichlet_recombination,
    }

    P = problem_class(
        **params,
    )

    u0 = P.u_exact(0)
    dt = 1e-1
    u = P.solve_system(u0, dt)

    if spectral_space:
        error = max(abs((P.M + dt * P.L) @ u.flatten() - P.M @ u0.flatten()))
        assert error < 1e-12, error

    iter_1 = P.work_counters[P.solver_type].niter
    assert iter_1 > 0

    u2 = P.solve_system(u0, u0=u, dt=dt)
    iter_2 = P.work_counters[P.solver_type].niter - iter_1

    assert iter_2 == 0, f'Did {iter_2} extra iterations after doing {iter_1} iterations first time'
    assert np.allclose(u, u2)


if __name__ == '__main__':
    test_initial_guess(False, True, 'bicgstab', True, True, 2**4)
