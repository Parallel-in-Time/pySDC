import pytest


@pytest.mark.base
def test_front_imex_vs_fully_implicit():
    import numpy as np
    from pySDC.implementations.problem_classes.AllenCahn_1D_FD import (
        allencahn_front_fullyimplicit,
        allencahn_front_semiimplicit,
        allencahn_front_finel,
    )

    problem_params = {
        'nvars': 31,
    }

    t0 = 0.0

    # for imex and fully-implicit test if right-hand side does match
    imex = allencahn_front_semiimplicit(**problem_params)
    full = allencahn_front_fullyimplicit(**problem_params)
    finel = allencahn_front_finel(**problem_params)

    u0 = imex.u_exact(t0)

    f_tmp = imex.eval_f(u0, t0)
    f_imex = f_tmp.impl + f_tmp.expl
    f_full = full.eval_f(u0, t0)

    assert np.allclose(
        f_imex, f_full
    ), "Evaluation of right-hand side in semi-implicit case and in fully-implicit case do not match!"

    # perform one time step and test the error
    dt = 1e-4
    args = {
        'rhs': u0,
        'factor': dt,
        'u0': u0,
        't': t0,
    }

    u_imex = imex.solve_system(**args)
    u_full = full.solve_system(**args)
    u_finel = finel.solve_system(**args)

    u_exact = imex.u_exact(t0 + dt)

    e_imex = abs(u_imex - u_exact) / abs(u_exact)
    e_full = abs(u_full - u_exact) / abs(u_exact)
    e_finel = abs(u_finel - u_exact) / abs(u_exact)

    assert e_imex < 8e-2, f"Error is too large in semi-implicit case! Got {e_imex}"
    assert e_full < 2e-3, f"Error is too large in fully-implicit case! Got {e_full}"
    assert e_finel < 2.5e-11, f"Error is too large in case of Finel's trick! Got {e_finel}"

    # check if number of right-hand side evaluations do match
    rhs_count_imex = imex.work_counters['rhs'].niter
    rhs_count_full = full.work_counters['rhs'].niter

    assert (
        rhs_count_imex == rhs_count_full
    ), "Number of right-hand side evaluations for semi-implicit vs fully-implicit do not match!"


@pytest.mark.base
def test_periodic_imex_vs_fully_implicit_vs_multi_implicit():
    import numpy as np
    from pySDC.implementations.problem_classes.AllenCahn_1D_FD import (
        allencahn_periodic_semiimplicit,
        allencahn_periodic_fullyimplicit,
        allencahn_periodic_multiimplicit,
    )

    problem_params = {
        'nvars': 32,
    }

    t0 = 0.0

    # for imex and fully-implicit test if right-hand side does match
    imex = allencahn_periodic_semiimplicit(**problem_params)
    full = allencahn_periodic_fullyimplicit(**problem_params)
    multi = allencahn_periodic_multiimplicit(**problem_params)

    u0 = imex.u_exact(t0)

    f_tmp_full = imex.eval_f(u0, t0)
    f_imex = f_tmp_full.impl + f_tmp_full.expl
    f_full = full.eval_f(u0, t0)
    f_tmp_multi = multi.eval_f(u0, t0)
    f_multi = f_tmp_multi.comp1 + f_tmp_multi.comp2

    assert np.allclose(
        f_imex, f_full
    ), "Evaluation of right-hand side in semi-implicit case and in fully-implicit case do not match!"
    assert np.allclose(
        f_full, f_multi
    ), "Evaluation of right-hand side in fully-implicit case and in multi-implicit case do not match!"

    # perform one time step and check errors
    dt = 1e-4
    args = {
        'rhs': u0,
        'factor': dt,
        'u0': u0,
        't': t0,
    }

    u_imex = imex.solve_system(**args)
    u_full = full.solve_system(**args)
    u_multi1 = multi.solve_system_1(**args)
    u_multi2 = multi.solve_system_2(**args)

    u_exact = imex.u_exact(t0 + dt)

    e_imex = abs(u_imex - u_exact) / abs(u_exact)
    e_full = abs(u_full - u_exact) / abs(u_exact)
    e_multi1 = abs(u_multi1 - u_exact) / abs(u_exact)
    e_multi2 = abs(u_multi2 - u_exact) / abs(u_exact)

    assert e_imex < 1e-2, f"Error is too large in semi-implicit case! Got {e_imex}"
    assert e_full < 1.2e-3, f"Error is too large in fully-implicit case! Got {e_full}"
    assert e_multi1 < 1e-2, f"Error is too large in multi-implicit case solving the Laplacian part! Got {e_multi1}"
    assert (
        e_multi2 < 1.2e-2
    ), f"Error is too large in multi-implicit case solving the part without the Laplacian! Got {e_multi2}"

    # check if number of right-hand side evaluations do match
    rhs_count_imex = imex.work_counters['rhs'].niter
    rhs_count_full = full.work_counters['rhs'].niter
    rhs_count_multi = multi.work_counters['rhs'].niter

    assert (
        rhs_count_imex == rhs_count_full
    ), "Number of right-hand side evaluations for semi-implicit vs fully-implicit do not match!"
    assert (
        rhs_count_full == rhs_count_multi
    ), "Number of right-hand side evaluations for fully-implicit vs multi-implicit do not match!"


@pytest.mark.base
@pytest.mark.parametrize('stop_at_nan', [True, False])
def test_capture_errors_and_warnings(caplog, stop_at_nan):
    """
    Test if errors and warnings are raised correctly.
    """
    import numpy as np
    from pySDC.core.Errors import ProblemError
    from pySDC.implementations.problem_classes.AllenCahn_1D_FD import (
        allencahn_front_fullyimplicit,
        allencahn_front_semiimplicit,
        allencahn_front_finel,
        allencahn_periodic_fullyimplicit,
        allencahn_periodic_semiimplicit,
        allencahn_periodic_multiimplicit,
    )

    newton_maxiter = 1
    problem_params = {
        'stop_at_nan': stop_at_nan,
        'newton_tol': 1e-13,
        'newton_maxiter': newton_maxiter,
    }

    full_front = allencahn_front_fullyimplicit(**problem_params)
    imex_front = allencahn_front_semiimplicit(**problem_params)
    finel_front = allencahn_front_finel(**problem_params)

    full_periodic = allencahn_periodic_fullyimplicit(**problem_params)
    imex_periodic = allencahn_periodic_semiimplicit(**problem_params)
    multi_periodic = allencahn_periodic_multiimplicit(**problem_params)

    t0 = 0.0
    dt = 1e-3

    u0_front = full_front.u_exact(t0)
    u0_periodic = full_periodic.u_exact(t0)

    args_front = {
        'rhs': u0_front,
        'factor': np.nan,
        'u0': u0_front,
        't': t0,
    }

    args_periodic = {
        'rhs': u0_periodic,
        'factor': np.nan,
        'u0': u0_periodic,
        't': t0,
    }

    if stop_at_nan:
        # test if ProblemError is raised
        with pytest.raises(ProblemError):
            full_front.solve_system(**args_front)
            imex_front.solve_system(**args_front)
            finel_front.solve_system(**args_front)

        with pytest.raises(ProblemError):
            full_periodic.solve_system(**args_periodic)
            imex_periodic.solve_system(**args_periodic)
            multi_periodic.solve_system_2(**args_periodic)

    else:
        # test if warnings are raised when nan values arise
        full_front.solve_system(**args_front)
        assert f'Newton got nan after {newton_maxiter} iterations...' in caplog.text
        assert 'Newton did not converge after 1 iterations, error is nan' in caplog.text
        assert (
            full_front.work_counters['newton'].niter == newton_maxiter
        ), 'Number of Newton iterations in fully-implicit front case does not match with maximum number of iterations!'
        caplog.clear()

        finel_front.solve_system(**args_front)
        assert f'Newton got nan after {newton_maxiter} iterations...' in caplog.text
        assert 'Newton did not converge after 1 iterations, error is nan' in caplog.text
        assert (
            finel_front.work_counters['newton'].niter == newton_maxiter
        ), "Number of Newton iterations in case of using Finel's trick does not match with maximum number of iterations!"
        caplog.clear()

        full_periodic.solve_system(**args_periodic)
        assert f'Newton got nan after {newton_maxiter} iterations...' in caplog.text
        assert 'Newton did not converge after 1 iterations, error is nan' in caplog.text
        assert (
            full_periodic.work_counters['newton'].niter == newton_maxiter
        ), 'Number of Newton iterations in fully-implicit periodic case does not match with maximum number of iterations!'
        caplog.clear()

        multi_periodic.solve_system_2(**args_periodic)
        assert f'Newton got nan after {newton_maxiter} iterations...' in caplog.text
        assert 'Newton did not converge after 1 iterations, error is nan' in caplog.text
        assert (
            multi_periodic.work_counters['newton'].niter == newton_maxiter
        ), 'Number of Newton iterations in multi-implicit periodic case does not match with maximum number of iterations!'
        caplog.clear()
