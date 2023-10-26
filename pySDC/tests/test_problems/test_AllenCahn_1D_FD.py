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

    assert np.allclose(f_imex, f_full), "Evaluation of right-hand side in semi-implicit case and in fully-implicit case do not match!"

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

    assert rhs_count_imex == rhs_count_full, "Number of right-hand side evaluations for semi-implicit vs fully-implicit do not match!"


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

    assert np.allclose(f_imex, f_full), "Evaluation of right-hand side in semi-implicit case and in fully-implicit case do not match!"
    assert np.allclose(f_full, f_multi), "Evaluation of right-hand side in fully-implicit case and in multi-implicit case do not match!"

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
    u_multi = multi.solve_system(**args)

    u_exact = imex.u_exact(t0 + dt)

    e_imex = abs(u_imex - u_exact) / abs(u_exact)
    e_full = abs(u_full - u_exact) / abs(u_exact)
    e_multi = abs(u_multi - u_exact) / abs(u_exact)

    assert e_imex < 1e-2, f"Error is too large in semi-implicit case! Got {e_imex}"
    assert e_full < 1.2e-3, f"Error is too large in fully-implicit case! Got {e_full}"
    assert e_multi < 1.2e-3, f"Error is too large in multi-implicit case! Got {e_multi}"

    # check if number of right-hand side evaluations do match
    rhs_count_imex = imex.work_counters['rhs'].niter
    rhs_count_full = full.work_counters['rhs'].niter
    rhs_count_multi = multi.work_counters['rhs'].niter

    assert rhs_count_imex == rhs_count_full, "Number of right-hand side evaluations for semi-implicit vs fully-implicit do not match!"
    assert rhs_count_full == rhs_count_multi, "Number of right-hand side evaluations for fully-implicit vs multi-implicit do not match!"

