r"""
Tests for :mod:`pySDC.implementations.sweeper_classes.delta_form`.

The delta form is an algebraic rewrite of the SDC sweep, so most of these assert that it changes no
answer -- against :class:`generic_implicit` and :class:`imex_1st_order`, for several
preconditioners, single- and multi-level, and under PFASST. The rest cover the three routes to the
correction equation: ``linear_implicit``, a problem's own ``solve_system_delta``, and the
substitution fallback.

The precision benefit the rewrite exists for is measured separately; what is asserted here is that
taking it costs no accuracy at backend precision.
"""

import numpy as np
import pytest

SWEEPER_PARAMS = {
    'quad_type': 'RADAU-RIGHT',
    'node_type': 'LEGENDRE',
    'num_nodes': 3,
    'QI': 'LU',
    'initial_guess': 'spread',
}

HEAT_PARAMS = {
    'nvars': 63,
    'nu': 1.0,
    'freq': 2,
    'bc': 'dirichlet-zero',
    'order': 2,
    'solver_type': 'direct',
}

AC_PARAMS = {
    'nvars': (32, 32),
    'eps': 0.04,
    'newton_maxiter': 100,
    'newton_tol': 1e-12,
    'lin_tol': 1e-12,
    'lin_maxiter': 200,
    'radius': 0.25,
}


def sweeper_params(**extra):
    params = dict(SWEEPER_PARAMS)
    params.update(extra)
    return params


def run(problem_class, problem_params, sweeper_class, sweeper_params, dt, nsteps, maxiter, num_procs=1, restol=-1):
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    description = {
        'problem_class': problem_class,
        'problem_params': problem_params,
        'sweeper_class': sweeper_class,
        'sweeper_params': sweeper_params,
        'level_params': {'restol': restol, 'dt': dt},
        'step_params': {'maxiter': maxiter},
    }
    controller = controller_nonMPI(num_procs=num_procs, controller_params={'logger_level': 30}, description=description)
    prob = controller.MS[0].levels[0].prob
    uend, stats = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=nsteps * dt)
    return uend, stats, prob


@pytest.mark.base
@pytest.mark.parametrize('QI', ['IE', 'LU', 'MIN-SR-S'])
@pytest.mark.parametrize('maxiter', [1, 3, 8])
def test_delta_form_matches_generic_implicit(QI, maxiter):
    """The delta form is an algebraic rewrite, so it must reproduce the standard sweep exactly."""
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit

    args = (heatNd_unforced, HEAT_PARAMS, sweeper_params(QI=QI), 1e-2, 2, maxiter)
    u_std, _, _ = run(args[0], args[1], generic_implicit, args[2], *args[3:])
    u_delta, _, _ = run(args[0], args[1], delta_implicit, args[2], *args[3:])
    assert abs(u_std - u_delta) < 1e-13, f'delta form deviates by {abs(u_std - u_delta):.3e}'


@pytest.mark.base
def test_zero_diagonal_preconditioner():
    """A Picard preconditioner has a zero diagonal, exercising the explicit branch."""
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit

    args = (heatNd_unforced, HEAT_PARAMS, sweeper_params(QI='PIC'), 1e-3, 1, 3)
    u_std, _, _ = run(args[0], args[1], generic_implicit, args[2], *args[3:])
    u_delta, _, _ = run(args[0], args[1], delta_implicit, args[2], *args[3:])
    assert abs(u_std - u_delta) < 1e-13


@pytest.mark.base
@pytest.mark.parametrize('precision', ['float32', 'float16'])
def test_correction_precision_storage(precision):
    """Storing the small quantities at reduced precision must not change the answer much."""
    import numpy as np
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit

    u64, _, prob = run(heatNd_unforced, HEAT_PARAMS, delta_implicit, sweeper_params(), 1e-2, 2, 6)
    u_red, _, _ = run(
        heatNd_unforced,
        HEAT_PARAMS,
        delta_implicit,
        sweeper_params(correction_precision=np.dtype(precision)),
        1e-2,
        2,
        6,
    )
    tolerance = 1e-10 if precision == 'float32' else 1e-3
    assert abs(u64 - u_red) < tolerance


@pytest.mark.base
def test_linear_implicit_reuses_stock_solve_system():
    """For a linear operator the stock solve_system already solves the correction equation."""
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit

    args = (heatNd_unforced, HEAT_PARAMS, 1e-2, 2, 6)
    u_std, _, _ = run(args[0], args[1], generic_implicit, sweeper_params(), *args[2:])
    u_delta, _, _ = run(args[0], args[1], delta_implicit, sweeper_params(linear_implicit=True), *args[2:])
    assert abs(u_std - u_delta) < 1e-12


def affine_heat():
    r"""
    The heat equation with a constant source, :math:`f(u) = Au + b`.

    Since :math:`f(w+\delta) - f(w) = A\delta` regardless of :math:`b`, the stock ``solve_system``
    solves the correction equation only once the affine part has been removed, which is what the
    sweeper does.
    """
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced

    class heat_affine(heatNd_unforced):
        B_CONST = 0.7

        def eval_f(self, u, t):
            f = super().eval_f(u, t)
            f[:] = np.asarray(f) + self.B_CONST
            return f

        def solve_system(self, rhs, factor, u0, t):
            identity = sp.eye(self.A.shape[0], format='csc')
            b = np.asarray(rhs, dtype=np.float64).reshape(-1) + factor * self.B_CONST
            me = self.dtype_u(self.init)
            me[:] = spsolve((identity - factor * self.A).tocsc(), b).reshape(self.nvars)
            return me

    return heat_affine


@pytest.mark.base
def test_linear_implicit_handles_affine_operator():
    """An affine operator needs the f(0, t) shift, which the sweeper subtracts."""
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit

    heat_affine = affine_heat()

    args = (heat_affine, HEAT_PARAMS, 1e-2, 2, 6)
    u_std, _, _ = run(args[0], args[1], generic_implicit, sweeper_params(), *args[2:])
    u_delta, _, _ = run(args[0], args[1], delta_implicit, sweeper_params(linear_implicit=True), *args[2:])
    assert abs(u_std - u_delta) < 1e-12


@pytest.mark.base
def test_mlsdc_tau_correction():
    """The residual must pick up the FAS tau term on coarse levels."""
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit

    results = {}
    for sweeper in [generic_implicit, delta_implicit]:
        description = {
            'problem_class': heatNd_unforced,
            'problem_params': dict(HEAT_PARAMS, nvars=[63, 31]),
            'sweeper_class': sweeper,
            'sweeper_params': sweeper_params(),
            'level_params': {'restol': -1, 'dt': 1e-2},
            'step_params': {'maxiter': 5},
            'space_transfer_class': mesh_to_mesh,
            'space_transfer_params': {'iorder': 2, 'rorder': 2},
        }
        controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
        prob = controller.MS[0].levels[0].prob
        results[sweeper], _ = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=2e-2)

    assert abs(results[generic_implicit] - results[delta_implicit]) < 1e-12


@pytest.mark.base
@pytest.mark.parametrize('num_procs', [2, 4])
def test_parallel_in_time(num_procs):
    """Nothing in the controller changes, so block-parallel runs must agree with the serial one."""
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit

    u_ref, _, _ = run(heatNd_unforced, HEAT_PARAMS, generic_implicit, sweeper_params(), 1e-2, 4, 20, restol=1e-11)
    u_par, _, _ = run(
        heatNd_unforced,
        HEAT_PARAMS,
        delta_implicit,
        sweeper_params(),
        1e-2,
        4,
        20,
        num_procs=num_procs,
        restol=1e-11,
    )
    assert abs(u_ref - u_par) < 1e-9


@pytest.mark.base
@pytest.mark.parametrize('maxiter', [1, 4, 10])
def test_delta_imex_matches_imex_1st_order(maxiter):
    """The IMEX splitting survives because the correction equation needs no Jacobian."""
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced
    from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
    from pySDC.implementations.sweeper_classes.delta_form import delta_imex_1st_order

    params = sweeper_params(QI='IE', QE='EE')
    args = (heatNd_forced, HEAT_PARAMS, params, 5e-2, 2, maxiter)
    u_std, _, _ = run(args[0], args[1], imex_1st_order, args[2], *args[3:])
    u_delta, _, _ = run(args[0], args[1], delta_imex_1st_order, args[2], *args[3:])
    assert abs(u_std - u_delta) < 1e-12


@pytest.mark.base
def test_delta_imex_correction_precision_and_linear_implicit():
    """Exercise the reduced-precision storage and the linear-implicit path for IMEX."""
    import numpy as np
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced
    from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
    from pySDC.implementations.sweeper_classes.delta_form import delta_imex_1st_order

    args = (heatNd_forced, HEAT_PARAMS, 5e-2, 2, 6)
    u_std, _, _ = run(args[0], args[1], imex_1st_order, sweeper_params(QI='IE', QE='EE'), *args[2:])
    u_red, _, _ = run(
        args[0],
        args[1],
        delta_imex_1st_order,
        sweeper_params(QI='IE', QE='EE', correction_precision=np.dtype('float32'), linear_implicit=True),
        *args[2:],
    )
    assert abs(u_std - u_red) < 1e-8


@pytest.mark.base
def test_solve_system_delta_path_is_preferred():
    """
    When the problem offers a correction solve, the sweeper must take it in preference to both
    other routes -- and still return what ``generic_implicit`` returns.

    The fixture solves the same system the fallback would, so only the dispatch is under test; a
    genuinely nonlinear correction equation, expanded analytically, is a problem-class matter.
    """
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit

    calls = {'n': 0}

    class heat_with_correction_solve(heatNd_unforced):
        def solve_system_delta(self, r, factor, base, f_base, t):
            """Solve ``d - factor * A d = r``, which is the correction equation here."""
            calls['n'] += 1
            identity = sp.eye(self.A.shape[0], format='csc')
            me = self.dtype_u(self.init)
            me[:] = spsolve((identity - factor * self.A).tocsc(), np.asarray(r).reshape(-1)).reshape(self.nvars)
            return me

    args = (HEAT_PARAMS, 1e-2, 2, 6)
    u_std, _, _ = run(heatNd_unforced, args[0], generic_implicit, sweeper_params(), *args[1:])
    u_delta, _, _ = run(heat_with_correction_solve, args[0], delta_implicit, sweeper_params(), *args[1:])

    assert calls['n'] > 0, 'solve_system_delta was never called'
    assert abs(u_std - u_delta) < 1e-12, f'the correction solve deviates by {abs(u_std - u_delta):.3e}'


@pytest.mark.base
def test_nonlinear_fallback_without_correction_solve():
    """Without a correction solve the sweeper falls back to the substitution, still exactly."""
    from pySDC.implementations.problem_classes.AllenCahn_2D_FD import allencahn_fullyimplicit
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit

    args = (allencahn_fullyimplicit, AC_PARAMS, 4e-4, 2, 8)
    u_std, _, _ = run(args[0], args[1], generic_implicit, sweeper_params(), *args[2:])
    u_delta, _, _ = run(args[0], args[1], delta_implicit, sweeper_params(), *args[2:])
    assert abs(u_std - u_delta) < 1e-11


def _multilevel_run(sweeper_class, sweeper_params_, nvars, num_procs, maxiter=6, restol=-1):
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh

    description = {
        'problem_class': heatNd_unforced,
        'problem_params': dict(HEAT_PARAMS, nvars=nvars),
        'sweeper_class': sweeper_class,
        'sweeper_params': sweeper_params_,
        'level_params': {'restol': restol, 'dt': 1e-2},
        'step_params': {'maxiter': maxiter},
        'space_transfer_class': mesh_to_mesh,
        'space_transfer_params': {'iorder': 2, 'rorder': 2},
    }
    controller = controller_nonMPI(num_procs=num_procs, controller_params={'logger_level': 30}, description=description)
    prob = controller.MS[0].levels[0].prob
    uend, _ = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=4e-2)
    return uend


@pytest.mark.base
@pytest.mark.parametrize('nvars', [[63, 31], [63, 31, 15]])
def test_mlsdc_multiple_levels(nvars):
    """MLSDC with two and three levels, i.e. recursive coarsening."""
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit

    u_std = _multilevel_run(generic_implicit, sweeper_params(), nvars, num_procs=1)
    u_delta = _multilevel_run(delta_implicit, sweeper_params(), nvars, num_procs=1)
    assert abs(u_std - u_delta) < 1e-12, f'{len(nvars)}-level MLSDC deviates'


@pytest.mark.base
@pytest.mark.parametrize('num_procs', [2, 4])
def test_pfasst(num_procs):
    """PFASST: multiple levels AND multiple steps in a block, which MLSDC alone does not cover."""
    import numpy as np
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit

    u_std = _multilevel_run(generic_implicit, sweeper_params(), [63, 31], num_procs=num_procs)
    u_delta = _multilevel_run(delta_implicit, sweeper_params(), [63, 31], num_procs=num_procs)
    assert abs(u_std - u_delta) < 1e-12, f'PFASST on {num_procs} steps deviates'

    reduced = _multilevel_run(
        delta_implicit,
        sweeper_params(correction_precision=np.dtype('float32')),
        [63, 31],
        num_procs=num_procs,
    )
    assert abs(u_std - reduced) < 1e-9, 'PFASST with reduced-precision corrections deviates'


@pytest.mark.base
def test_correction_scaling_keeps_the_error_relative():
    """
    The scaling's job: a stored correction must round-trip with a *relative* error, however small it
    gets.

    Without it, half precision has almost no mantissa left below its smallest normal of 6.1e-5 --
    1.3e-2 relative at 1e-6, 1.9e-1 at 1e-7 -- so a correction turns to noise exactly when it starts
    to matter. Dividing by the residual's own magnitude first is block floating point, and keeps the
    error at the format's epsilon wherever the correction happens to be.
    """
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit

    prob = heatNd_unforced(**dict(HEAT_PARAMS, nvars=127))
    sweeper = delta_implicit.__new__(delta_implicit)
    sweeper._work_dtype = np.dtype('float16')

    for magnitude in (1e0, 1e-4, 1e-8, 1e-12):
        value = prob.dtype_u(prob.init)
        value[:] = magnitude * np.sin(np.linspace(0, 3, prob.nvars[0]))
        sweeper._set_work_scale([value])
        back = sweeper._to_backend(prob, sweeper._to_work(prob, value))
        relative = abs(back - value) / abs(value)
        assert relative < 1e-3, f'at magnitude {magnitude:.0e} the round trip lost {relative:.1e}'

    # and the control: without a scale, the same round trip collapses once below the smallest normal
    sweeper._work_scale = 1.0
    value = prob.dtype_u(prob.init)
    value[:] = 1e-12 * np.sin(np.linspace(0, 3, prob.nvars[0]))
    back = sweeper._to_backend(prob, sweeper._to_work(prob, value))
    assert abs(back - value) / abs(value) > 0.5, 'the unscaled round trip was supposed to fail here'


@pytest.mark.base
def test_float16_corrections_work_once_scaled():
    """
    Half-precision correction storage on a *fine* level, which needs the scaling to work at all.

    Unscaled it stalls at 5.9e-05, which is float16's smallest normal: the delta form drives the
    correction towards zero on purpose and half precision has almost no mantissa left down there.
    Scaled, it reaches the full float64 floor at one extra iteration. Single precision is free.

    The level's *state* is a different matter and stays float64 -- see
    :func:`test_float16_state_still_does_not_work`.
    """
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit

    def measure(prob_extra=None, **sweep_extra):
        description = {
            'problem_class': heatNd_unforced,
            'problem_params': dict(HEAT_PARAMS, nvars=127, **(prob_extra or {})),
            'sweeper_class': delta_implicit,
            'sweeper_params': sweeper_params(linear_implicit=True, **sweep_extra),
            'level_params': {'restol': -1, 'dt': 1e-1},
            'step_params': {'maxiter': 30},
        }
        controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
        prob = controller.MS[0].levels[0].prob
        uend, stats = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=1e-1)
        residuals = [value for _, value in get_sorted(stats, type='residual_post_iteration', sortby='iter')]
        hit = next((i + 1 for i, value in enumerate(residuals) if value < 1e-11), None)
        return hit, min(residuals), np.asarray(uend, dtype=float)

    baseline, floor, reference = measure()
    assert floor < 1e-12, 'the baseline did not converge'

    for precision in (np.float32, np.float16):
        hit, floor, uend = measure(correction_precision=precision)
        assert hit is not None and hit <= baseline + 1, f'{precision.__name__} corrections cost {hit} iterations'
        assert floor < 1e-12, f'{precision.__name__} corrections floored at {floor:.2e}'
        assert np.max(np.abs(uend - reference)) < 1e-13


@pytest.mark.base
@pytest.mark.parametrize('precision,epsilon', [('float32', 1.2e-7), ('float16', 9.8e-4)])
def test_stored_corrections_really_cost_the_format(precision, epsilon):
    """
    ``correction_precision`` has to store at the format and lose the format's precision doing it.

    Since the scaling was added, ``_to_work`` divides and ``_to_backend`` multiplies, and a mistake
    in either could make the pair a no-op that quietly stores at full precision. So: check the dtype
    of what is stored, that the round trip costs the format's epsilon, and that the scale actually
    follows the residual down rather than sitting at one.
    """
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit

    stored, losses, scales = [], [], []

    class watched(delta_implicit):
        def _to_work(self, prob, value):
            out = super()._to_work(prob, value)
            stored.append(str(np.asarray(out).dtype))
            before = np.asarray(value, dtype=np.float64)
            after = np.asarray(self._to_backend(prob, out), dtype=np.float64)
            losses.append(np.max(np.abs(after - before)) / max(np.max(np.abs(before)), 1e-300))
            scales.append(self._work_scale)
            return out

    description = {
        'problem_class': heatNd_unforced,
        'problem_params': dict(HEAT_PARAMS, nvars=127),
        'sweeper_class': watched,
        'sweeper_params': sweeper_params(linear_implicit=True, correction_precision=np.dtype(precision)),
        'level_params': {'restol': -1, 'dt': 1e-1},
        'step_params': {'maxiter': 20},
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    prob = controller.MS[0].levels[0].prob
    controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=1e-1)

    assert set(stored) == {precision}, f'corrections were stored as {set(stored)}'
    median = float(np.median(losses))
    assert epsilon / 100 < median < epsilon, f'the round trip cost {median:.1e}, not the format epsilon'
    assert min(scales) < 1e-10 < max(scales), f'the scale did not follow the residual down: {min(scales):.1e}'


@pytest.mark.base
def test_a_handed_down_residual_is_used_and_advanced():
    r"""
    The sweeper computes its own residual unless a transfer hands one down.

    A hierarchy that reformulates the coarse level hands one down instead of letting the level
    rebuild it out of :math:`\mathcal{O}(1)` state -- that transfer is a separate thing, so the
    handing down is stubbed here. What is under test is the sweeper's half of the contract: use
    ``eps_in`` in place of the residual it would have computed, advance it by
    :math:`\varepsilon \leftarrow \varepsilon - \delta + \Delta t (Q \Delta f)` after the sweep, and
    bank the corrections for the transfer to take back.

    The recursion is checked against the residual recomputed from scratch, which is the only thing
    that says the advance is right rather than merely self-consistent.
    """
    import numpy as np
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit

    drift, banked = [], []

    class handed_down(delta_implicit):
        def update_nodes(self):
            if self.eps_in is None:
                # stand in for a transfer: hand the level the residual it was about to compute
                self.eps_in = self._residual_nodes()
            super().update_nodes()
            rebuilt = delta_implicit._residual_nodes(self)
            drift.append(max(float(np.max(np.abs(np.asarray(a - b)))) for a, b in zip(rebuilt, self.eps_in)))
            banked.append(self.delta_acc is not None)

    args = (heatNd_unforced, HEAT_PARAMS, sweeper_params(linear_implicit=True), 1e-2, 1, 6)
    reference, _, _ = run(args[0], args[1], delta_implicit, args[2], *args[3:])
    uend, _, _ = run(args[0], args[1], handed_down, args[2], *args[3:])

    assert len(drift) == 6, f'the sweep ran {len(drift)} times'
    assert all(banked), 'the corrections were not banked for a transfer to take back'
    assert max(drift) < 1e-13, f'the tracked residual drifted from the real one by {max(drift):.3e}'
    assert abs(uend - reference) < 1e-13, f'carrying the residual moved the answer by {abs(uend - reference):.3e}'
