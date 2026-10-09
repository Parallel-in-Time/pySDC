"""The inexact-solve wrapper, Van der Pol's correction solve, and the eta sweep's plumbing."""

import numpy as np
import pytest


@pytest.mark.base
def test_inexact_spoils_by_exactly_eta():
    """The returned solve is off by eta times its maximum, in the maximum norm; eta=None is exact."""
    from pySDC.projects.DeltaSDC.inexact import inexact
    from pySDC.projects.DeltaSDC.problems import heat_delta

    params = {'nvars': 63, 'nu': 1.0, 'bc': 'dirichlet-zero'}
    exact, spoiled = heat_delta(**params), inexact(heat_delta)(eta=1e-3, **params)
    rhs = exact.u_exact(0.0)
    x, y = (np.asarray(p.solve_system(rhs, 1e-2, rhs, 0.0)) for p in (exact, spoiled))
    assert np.isclose(np.abs(y - x).max(), 1e-3 * np.abs(x).max())
    assert np.array_equal(np.asarray(inexact(heat_delta)(**params).solve_system(rhs, 1e-2, rhs, 0.0)), x)


@pytest.mark.base
def test_vanderpol_delta():
    """The increment is exact without cancellation, and delta-form SDC reproduces generic_implicit."""
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.projects.DeltaSDC.problems import vanderpol_delta

    prob = vanderpol_delta(mu=10.0)
    w, d = prob.dtype_u(prob.init), prob.dtype_u(prob.init)
    w[:], d[:] = [1.3, -0.7], [1e-3, -2e-3]
    increment = np.asarray(prob.eval_f_increment(w, d, 0.0))
    assert np.abs(increment - np.asarray(prob.eval_f(w + d, 0.0) - prob.eval_f(w, 0.0))).max() < 1e-13

    ends = []
    for sweeper in [generic_implicit, delta_implicit]:
        description = {
            'problem_class': vanderpol_delta,
            'problem_params': {'mu': 10.0, 'newton_tol': 1e-14},
            'sweeper_class': sweeper,
            'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'LU'},
            'level_params': {'restol': 1e-12, 'dt': 0.05},
            'step_params': {'maxiter': 50},
        }
        controller = controller_nonMPI(1, {'logger_level': 40}, description)
        uend, _ = controller.run(u0=controller.MS[0].levels[0].prob.u_exact(0.0), t0=0, Tend=0.5)
        ends.append(np.asarray(uend))
    assert np.abs(ends[0] - ends[1]).max() < 1e-10


@pytest.mark.base
def test_eta_sweep_runs():
    """Two configurations, three etas: an exact run that converges, a contraction rate, an eta* and a prediction."""
    import os

    from pySDC.projects.DeltaSDC.eta_spec import cases, main

    specs = [c for c in cases() if c['label'] in ('heat dt=0.1', 'Van der Pol mu=10')]
    for spec, result in main(specs, etas=[1e-2, 1e-5, 1e-8], seeds=(0,)):
        assert np.isfinite(result['exact']) and 0 < result['rho'] < 1, spec['label']
        assert result['eta_star'] is not None, spec['label']
        assert result['predicted']['eta_star'] > 0, spec['label']
    assert os.path.isfile('data/eta_spec.png')


@pytest.mark.base
def test_round_to_formats():
    """Rounding relative to the maximum is accurate to the format's unit roundoff, and nothing more."""
    from pySDC.projects.DeltaSDC.inexact import FORMATS, round_to

    x = np.random.default_rng(0).standard_normal(1000) * 1e-9
    for fmt, u in FORMATS.items():
        error = np.abs(round_to(x, fmt) - x).max() / np.abs(x).max()
        assert 0.1 * u < error <= u, fmt


@pytest.mark.base
def test_schedule_holds():
    """The per-iteration schedule costs at most one iteration, with genuinely low-precision solves."""
    from pySDC.projects.DeltaSDC.eta_spec import cases
    from pySDC.projects.DeltaSDC.genuine import genuine
    from pySDC.projects.DeltaSDC.precision_theory import iterations, record_run, schedule

    spec = next(c for c in cases() if c['label'] == 'advection CFL=2')
    formats = schedule(spec)
    assert formats[0] == 'fp64' and formats[-1] == 'fp16', formats
    exact = iterations(record_run(spec)['history'], spec['restol'])
    params = dict(spec['problem_params'], schedule=tuple(formats), nodes=3)
    run = record_run(spec, problem_class=genuine(spec['problem_class']), problem_params=params, maxiter=exact + 3)
    assert iterations(run['history'], spec['restol']) <= exact + 1


@pytest.mark.base
def test_genuine_lu_is_backward_stable():
    """The LU really runs in the format: backward error about u, forward error up to about kappa u."""
    from pySDC.projects.DeltaSDC.genuine import lu, lu_solve
    from pySDC.projects.DeltaSDC.inexact import FORMATS

    n = 64
    B = np.eye(n) + 100.0 * (2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))
    b = np.random.default_rng(0).standard_normal(n)
    b /= np.abs(b).max()
    for fmt in ('fp32', 'fp16', 'bf16'):
        y = lu_solve(lu(B, fmt), b, fmt)
        backward = np.abs(B @ y - b).max() / (np.abs(B).sum(1).max() * np.abs(y).max())
        assert 0.01 * FORMATS[fmt] < backward < 4 * FORMATS[fmt], fmt
