import pytest

AC_PARAMS = {
    'nvars': (32, 32),
    'eps': 0.04,
    'newton_maxiter': 100,
    'newton_tol': 1e-12,
    'lin_tol': 1e-12,
    'lin_maxiter': 200,
    'radius': 0.25,
}


@pytest.mark.base
def test_rejects_unsupported_nu():
    """The analytic increment is derived for nu=2 only."""
    from pySDC.projects.DeltaSDC.problems import allencahn_delta

    with pytest.raises(NotImplementedError, match='nu=2'):
        allencahn_delta(nu=1, **AC_PARAMS)


@pytest.mark.base
def test_increment_is_exact_and_cancellation_free():
    """The analytic increment must equal f(w+d) - f(w) and vanish linearly with d."""
    import numpy as np
    from pySDC.projects.DeltaSDC.problems import allencahn_delta

    prob = allencahn_delta(**AC_PARAMS)
    base = np.asarray(prob.u_exact(0.0)).reshape(-1)

    previous = None
    for scale in [1e-2, 1e-4, 1e-6]:
        delta = scale * np.sin(np.arange(base.size) * 0.01)
        analytic = prob._increment(base, delta)
        reference = np.asarray(prob.eval_f(prob.u_exact(0.0) * 0 + (base + delta).reshape(prob.nvars), 0.0)).reshape(
            -1
        ) - np.asarray(prob.eval_f(base.reshape(prob.nvars), 0.0)).reshape(-1)
        assert np.max(np.abs(analytic - reference)) < 1e-8 * max(np.max(np.abs(reference)), 1.0)

        magnitude = np.max(np.abs(analytic))
        if previous is not None:
            assert magnitude < previous, 'increment must shrink with the correction'
        previous = magnitude


@pytest.mark.base
def test_jacobian_is_assembled_at_working_precision():
    """The system matrix must be built directly at the working precision, not cast afterwards."""
    import numpy as np
    from pySDC.projects.DeltaSDC.problems import allencahn_delta

    for precision, expected in [(None, np.dtype('float64')), (np.float32, np.dtype('float32'))]:
        prob = allencahn_delta(solve_precision=precision, **AC_PARAMS)
        state = np.asarray(prob.u_exact(0.0), dtype=expected).reshape(-1)
        jacobian = prob._jacobian(state, expected.type(1e-4))
        assert jacobian.dtype == expected


@pytest.mark.base
@pytest.mark.parametrize('precision', [None, 'float32'])
def test_solve_system_delta_solves_the_correction_equation(precision):
    """The returned correction must satisfy d - alpha [f(w+d) - f(w)] = r."""
    import numpy as np
    from pySDC.projects.DeltaSDC.precision import tol_floor
    from pySDC.projects.DeltaSDC.problems import allencahn_delta

    tol = tol_floor(precision)
    prob = allencahn_delta(solve_precision=precision, **dict(AC_PARAMS, lin_tol=tol, newton_tol=1e-14))

    base = prob.u_exact(0.0)
    rhs = prob.dtype_u(prob.init)
    rhs[:] = 1e-4 * np.sin(np.arange(prob.nvars[0] * prob.nvars[1]) * 0.01).reshape(prob.nvars)
    factor = 1e-5

    delta = prob.solve_system_delta(rhs, factor, base, prob.eval_f(base, 0.0), 0.0)

    d = np.asarray(delta).reshape(-1)
    w = np.asarray(base).reshape(-1)
    residual = d - factor * np.asarray(prob._increment(w.astype(prob._work_dtype), d.astype(prob._work_dtype)))
    residual = residual - np.asarray(rhs).reshape(-1)

    assert np.max(np.abs(residual)) < 10 * tol * max(np.max(np.abs(np.asarray(rhs))), 1e-30)
    assert np.max(np.abs(d)) > 0.0, 'correction must be non-trivial'


@pytest.mark.base
def test_reduced_precision_solve_matches_full_precision():
    """The correction solve at float32 must agree with the float64 one to fp32 relative accuracy."""
    import numpy as np
    from pySDC.projects.DeltaSDC.precision import tol_floor
    from pySDC.projects.DeltaSDC.problems import allencahn_delta

    tol = tol_floor(np.float32)
    results = {}
    for precision in [None, np.float32]:
        prob = allencahn_delta(solve_precision=precision, **dict(AC_PARAMS, lin_tol=tol, newton_tol=1e-14))
        base = prob.u_exact(0.0)
        rhs = prob.dtype_u(prob.init)
        rhs[:] = 1e-4 * np.sin(np.arange(prob.nvars[0] * prob.nvars[1]) * 0.01).reshape(prob.nvars)
        results[precision] = np.asarray(prob.solve_system_delta(rhs, 1e-5, base, prob.eval_f(base, 0.0), 0.0))

    scale = np.max(np.abs(results[None]))
    assert np.max(np.abs(results[None] - results[np.float32])) < 1e-4 * scale


@pytest.mark.base
def test_work_counters_are_advanced():
    """The correction solve must report its linear and nonlinear work."""
    import numpy as np
    from pySDC.projects.DeltaSDC.problems import allencahn_delta

    prob = allencahn_delta(**dict(AC_PARAMS, lin_tol=1e-10, newton_tol=1e-13))
    base = prob.u_exact(0.0)
    rhs = prob.dtype_u(prob.init)
    rhs[:] = 1e-3 * np.sin(np.arange(prob.nvars[0] * prob.nvars[1]) * 0.01).reshape(prob.nvars)

    prob.solve_system_delta(rhs, 1e-4, base, prob.eval_f(base, 0.0), 0.0)

    assert prob.work_counters['newton'].niter > 0
    assert prob.work_counters['linear'].niter > 0


@pytest.mark.base
def test_demo_runs():
    """The shipped demo must execute and reproduce the reference to round-off."""
    from pySDC.projects.DeltaSDC.run_demo import main

    results = main()
    reference = results['generic_implicit (fp64)']['uend']
    for label, result in results.items():
        assert abs(result['uend'] - reference) < 1e-10, f'{label} deviates'
        assert result['niter'] == results['delta-form (fp64)']['niter']


@pytest.mark.base
def test_operator_norm_tracks_resolution_and_epsilon():
    """The conditioning estimate must grow with the resolution and with 1/eps^2."""
    from pySDC.projects.DeltaSDC.problems import allencahn_delta

    coarse = allencahn_delta(**dict(AC_PARAMS, nvars=(32, 32)))
    fine = allencahn_delta(**dict(AC_PARAMS, nvars=(64, 64)))
    sharp = allencahn_delta(**dict(AC_PARAMS, nvars=(32, 32), eps=0.02))

    assert fine._operator_norm > coarse._operator_norm
    assert sharp._operator_norm > coarse._operator_norm
    assert coarse._operator_norm > 0.0


@pytest.mark.base
def test_stiff_configuration_does_not_stall_in_reduced_precision():
    """Regression: a fixed eps-multiple floor made this configuration run to maxiter in fp32."""
    import numpy as np
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.projects.DeltaSDC.precision import tol_floor
    from pySDC.projects.DeltaSDC.problems import allencahn_delta
    from pySDC.projects.DeltaSDC.sweepers import delta_implicit

    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'node_type': 'LEGENDRE',
        'num_nodes': 3,
        'QI': 'LU',
        'initial_guess': 'spread',
    }
    counts = {}
    for precision in [None, np.float32]:
        params = dict(
            AC_PARAMS,
            nvars=(64, 64),
            lin_maxiter=500,
            solve_precision=precision,
            lin_tol=tol_floor(np.float32),
            newton_tol=1e-30,
        )
        description = {
            'problem_class': allencahn_delta,
            'problem_params': params,
            'sweeper_class': delta_implicit,
            'sweeper_params': sweeper_params,
            'level_params': {'restol': 1e-9, 'dt': 4e-3},
            'step_params': {'maxiter': 30},
        }
        controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
        prob = controller.MS[0].levels[0].prob
        uend, _ = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=8e-3)
        counts[precision] = (prob.work_counters['newton'].niter, uend)

    newton32, u32 = counts[np.float32]
    newton64, u64 = counts[None]

    # Before the conditioning-aware floor this ratio was ~39x, because the reduced-precision solve
    # ran to newton_maxiter on every call. Stopping earlier than fp64 is fine and expected: the
    # floor for float32 is legitimately looser.
    assert newton32 <= 2 * newton64, (
        f'reduced precision needed {newton32} Newton iterations against {newton64} at full '
        f'precision, i.e. it stalled'
    )
    assert abs(u32 - u64) < 1e-9, 'reduced precision changed the result'


@pytest.mark.base
def test_warns_when_the_correction_solve_hits_maxiter(caplog):
    """Exiting on maxiter rather than on the tolerance must be reported, not silent."""
    import logging
    import numpy as np
    from pySDC.projects.DeltaSDC.problems import allencahn_delta

    prob = allencahn_delta(**dict(AC_PARAMS, newton_maxiter=1, lin_tol=1e-10, newton_tol=1e-30))
    base = prob.u_exact(0.0)
    rhs = prob.dtype_u(prob.init)
    rhs[:] = 1e-2 * np.sin(np.arange(prob.nvars[0] * prob.nvars[1]) * 0.01).reshape(prob.nvars)

    with caplog.at_level(logging.WARNING, logger='problem'):
        prob.solve_system_delta(rhs, 1e-4, base, prob.eval_f(base, 0.0), 0.0)

    assert any('newton_maxiter' in record.message for record in caplog.records)
