"""
Tests for the precision cascade.

The load-bearing one is :func:`test_initial_value_is_never_rounded`: the cascade's failure mode is
not a diverging run but a converging one that returns the wrong answer, so a residual-based check
would not catch it.
"""

import pytest

SWEEPER_PARAMS = {
    'quad_type': 'RADAU-RIGHT',
    'node_type': 'LEGENDRE',
    'num_nodes': 3,
    'QI': 'LU',
    'initial_guess': 'spread',
    'linear_implicit': True,
}

HEAT_PARAMS = {
    'nu': 1.0,
    'freq': 2,
    'bc': 'dirichlet-zero',
    'order': 2,
    'solver_type': 'direct',
}

LADDER = ('float16', 'float32', None)


def build(multilevel, **extra):
    """Build a controller on the heat equation, single- or multi-level."""
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
    from pySDC.projects.DeltaSDC.cascade import delta_implicit_cascade
    from pySDC.projects.DeltaSDC.mlsdc import delta_transfer
    from pySDC.projects.DeltaSDC.problems import heat_delta

    description = {
        'problem_class': heat_delta,
        'problem_params': dict(HEAT_PARAMS, nvars=[127, 63] if multilevel else [127]),
        'sweeper_class': delta_implicit_cascade,
        'sweeper_params': dict(SWEEPER_PARAMS, **extra),
        'level_params': {'restol': 1e-11, 'dt': 1e-1},
        'step_params': {'maxiter': 40},
        'space_transfer_class': mesh_to_mesh,
        'space_transfer_params': {'iorder': 4, 'rorder': 2},
        'base_transfer_class': delta_transfer,
    }
    return controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)


def run(multilevel, **extra):
    """Run one step and return the end value, the iteration count and the format history."""
    from pySDC.helpers.stats_helper import get_sorted

    controller = build(multilevel, **extra)
    prob = controller.MS[0].levels[0].prob
    uend, stats = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=1e-1)
    residuals = [value for _, value in get_sorted(stats, type='residual_post_iteration', sortby='iter')]
    return uend, len(residuals), controller.MS[0].levels[0].sweep.cascade_history


@pytest.mark.base
@pytest.mark.parametrize('multilevel', [False, True])
def test_cascade_reaches_the_full_precision_answer(multilevel):
    """Walking up the ladder must end where a float64 run ends, within one iteration of it."""
    reference, baseline, _ = run(multilevel)
    uend, count, _ = run(multilevel, state_cascade=LADDER)

    assert abs(uend - reference) < 1e-12, f'the cascade moved the answer by {abs(uend - reference):.3e}'
    assert count <= baseline + 1, f'the cascade cost {count - baseline} iterations'


@pytest.mark.base
@pytest.mark.parametrize('multilevel', [False, True])
def test_cascade_actually_walks_the_ladder(multilevel):
    """
    Every other test here would pass just as well if the cascade never left float64.

    So check that each rung is used and that the walk is monotone -- a policy allowed to step back
    down would convert the level every sweep instead of at most twice.
    """
    _, _, history = run(multilevel, state_cascade=LADDER, cascade_safety=10.0)

    assert set(history) == {'float16', 'float32', None}, f'not every rung was used: {history}'
    positions = [LADDER.index(fmt) for fmt in history]
    assert positions == sorted(positions), f'the cascade stepped back down: {history}'
    assert history[-1] is None, 'the run has to end at backend precision'


@pytest.mark.base
def test_initial_value_is_never_rounded():
    """
    The cascade may only lower the precision of quantities the iteration later corrects.

    ``u[0]`` is not one of them: it is the step's initial value, no sweep touches it, and raising the
    precision afterwards does not undo a rounding. Getting this wrong does not diverge -- it
    converges to a residual of 2.7e-12 and an answer 4.0e-06 out -- so it is checked directly.
    """
    controller = build(False, state_cascade=LADDER, cascade_safety=10.0)
    level = controller.MS[0].levels[0]
    exact = level.prob.u_exact(0.0)
    controller.run(u0=exact, t0=0.0, Tend=1e-1)

    assert abs(level.u[0] - exact) == 0.0, 'the cascade rounded the initial value'
    # and the run really did drop below backend precision, or the check above is vacuous
    assert 'float16' in level.sweep.cascade_history


@pytest.mark.base
def test_ladder_must_be_a_tuple():
    """
    pySDC spreads a list-valued parameter one entry per level.

    A list is therefore taken apart before the sweeper sees it, arriving as a single format name,
    and indexing into that picks out characters rather than formats -- 'float16'[1] is 'l', which
    numpy reads as int64. Refused rather than silently misread.
    """
    with pytest.raises(ValueError, match='must be a tuple of formats'):
        run(False, state_cascade=['float16', 'float32', None])


@pytest.mark.base
def test_cascade_leaves_coarse_levels_alone():
    """
    The cascade is for the finest level, and must not undo a coarse level's fixed setting.

    A coarse level is a preconditioner: its requirement is relative, so it belongs at a fixed low
    precision rather than on a ladder, and there is nothing for it to climb towards. Left
    unrestricted the cascade climbs there too and returns the coarse level to backend precision
    after a few sweeps, silently throwing away the setting.
    """
    import numpy as np

    controller = build(True, level_precision=[None, np.float16], state_cascade=LADDER, cascade_safety=10.0)
    prob = controller.MS[0].levels[0].prob
    controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=1e-1)

    fine, coarse = controller.MS[0].levels
    assert fine.sweep.cascade_history, 'the finest level did not cascade'
    assert set(fine.sweep.cascade_history) != {None}, 'the finest level never left backend precision'
    assert not coarse.sweep.cascade_history, 'the cascade ran on a coarse level'
    # and the coarse level really is still at its fixed format
    values = np.asarray(coarse.u[1])
    assert np.array_equal(values, values.astype(np.float16)), 'the coarse level left float16'
