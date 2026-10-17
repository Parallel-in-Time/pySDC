r"""
What the delta-form hierarchy buys: a coarse level below backend precision.

The hierarchy itself is ordinary pySDC and is tested in
``pySDC/tests/test_transfer_classes/test_BaseTransferDelta.py``. Here it is put to work with
``level_precision``, and paired with the controls that must break -- the stock hierarchy at the same
reduced precision, the finest level reduced, and the increment formed by subtraction. Without those
rows the rest measures nothing.

The last test replaces the format by the accuracy a node-local solve actually *delivers*, which is
all the iteration can see.
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
    'nu': 1.0,
    'freq': 2,
    'bc': 'dirichlet-zero',
    'order': 2,
    'solver_type': 'direct',
}


def sweeper_params(**extra):
    params = dict(SWEEPER_PARAMS, linear_implicit=True)
    params.update(extra)
    return params


def build(nvars, sweeper_class, transfer_class, num_procs=1, maxiter=6, dt=1e-1, problem_class=None, **extra):
    """Build a controller on the heat equation with the given hierarchy."""
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
    from pySDC.projects.DeltaSDC.problems import heat_delta

    description = {
        'problem_class': problem_class or heat_delta,
        'problem_params': dict(HEAT_PARAMS, nvars=nvars),
        'sweeper_class': sweeper_class,
        'sweeper_params': sweeper_params(**extra),
        'level_params': {'restol': -1, 'dt': dt},
        'step_params': {'maxiter': maxiter},
        'space_transfer_class': mesh_to_mesh,
        'space_transfer_params': {'iorder': 4, 'rorder': 2},
        'base_transfer_class': transfer_class,
    }
    return controller_nonMPI(num_procs=num_procs, controller_params={'logger_level': 30}, description=description)


def run(*args, nsteps=1, dt=1e-1, **kwargs):
    """Run and return the end value."""
    controller = build(*args, dt=dt, **kwargs)
    prob = controller.MS[0].levels[0].prob
    uend, _ = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=nsteps * dt)
    return uend


def floor(*args, nsteps=1, dt=1e-1, **kwargs):
    """Run and return the residual floor, which is what a stalling iteration shows."""
    from pySDC.helpers.stats_helper import get_sorted

    controller = build(*args, dt=dt, **kwargs)
    prob = controller.MS[0].levels[0].prob
    _, stats = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=nsteps * dt)
    return min(value for _, value in get_sorted(stats, type='residual_post_iteration', sortby='iter'))


@pytest.mark.base
@pytest.mark.parametrize('precision', ['float32', 'float16'])
@pytest.mark.parametrize('nvars', [[127, 63], [127, 63, 31]])
def test_reduced_precision_coarse_levels(precision, nvars):
    """Every level below the finest can drop precision without moving the answer."""
    import numpy as np
    from pySDC.projects.DeltaSDC.mlsdc import delta_implicit_rounded, delta_transfer

    reference = run(nvars, delta_implicit_rounded, delta_transfer)
    reduced = run(
        nvars,
        delta_implicit_rounded,
        delta_transfer,
        level_precision=[None] + [np.dtype(precision)] * (len(nvars) - 1),
    )
    assert abs(reference - reduced) < 1e-13, f'{precision} coarse levels move the answer'


@pytest.mark.base
@pytest.mark.parametrize('precision', ['float32', 'float16'])
def test_stock_hierarchy_breaks_at_reduced_precision(precision):
    """
    The control: the same reduced-precision coarse level, with the two cancellations left in.

    Without this the test above says nothing -- it would pass just as well if the rounding were
    never applied.
    """
    import numpy as np
    from pySDC.projects.DeltaSDC.mlsdc import (
        delta_implicit_rounded,
        delta_transfer,
        rounding_transfer,
    )

    # measured on the residual floor rather than the end value: the injected noise is grid-scale and
    # an LU sweep on a stiff operator largely cleans it out of the *state* again, so what a stalling
    # iteration shows is a residual that stops falling
    reference = floor([127, 63], delta_implicit_rounded, delta_transfer, maxiter=25)
    stock = floor(
        [127, 63], delta_implicit_rounded, rounding_transfer, level_precision=[None, np.dtype(precision)], maxiter=25
    )
    assert reference < 1e-12, f'the delta hierarchy did not converge, floor {reference:.3e}'
    assert stock > 1e-11, f'the {precision} control floored at {stock:.3e}, so it controls nothing'


@pytest.mark.base
def test_fine_level_precision_binds():
    """The other control: the finest level is the one place precision cannot be reduced."""
    import numpy as np
    from pySDC.projects.DeltaSDC.mlsdc import delta_implicit_rounded, delta_transfer

    reference = floor([127, 63], delta_implicit_rounded, delta_transfer, maxiter=25)
    reduced = floor([127, 63], delta_implicit_rounded, delta_transfer, level_precision=np.dtype('float32'), maxiter=25)
    assert reference < 1e-12, f'the reference did not converge, floor {reference:.3e}'
    assert reduced > 1e-11, f'reducing the fine level floored at {reduced:.3e}, so it controls nothing'


@pytest.mark.base
def test_analytic_increment_is_needed_below_backend_precision():
    """
    Without ``eval_f_increment`` the sweeper subtracts two stored right-hand sides.

    That cancellation carries the operator norm, so it is the first thing to break when a level's
    precision drops -- before the residual, and before the node-local solve.
    """
    import numpy as np
    from pySDC.projects.DeltaSDC.mlsdc import delta_implicit_rounded, delta_transfer
    from pySDC.projects.DeltaSDC.problems import heat_no_increment

    # every GenericNDimFinDiff problem supplies the increment now, so the control needs a class that
    # hides it again rather than a stock problem that happens not to have one
    assert not hasattr(heat_no_increment(nvars=31, **HEAT_PARAMS), 'eval_f_increment')

    reference = floor([127, 63], delta_implicit_rounded, delta_transfer, maxiter=25)
    subtracted = floor(
        [127, 63],
        delta_implicit_rounded,
        delta_transfer,
        problem_class=heat_no_increment,
        level_precision=[None, np.dtype('float32')],
        maxiter=25,
    )
    assert reference < 1e-12, f'the reference did not converge, floor {reference:.3e}'
    assert subtracted > 1e-11, f'the subtraction floored at {subtracted:.3e}, so it controls nothing'


@pytest.mark.base
def test_level_precision_refuses_what_it_cannot_round():
    """An emulation that quietly did nothing would report a reduced-precision result that never was."""
    import numpy as np
    from pySDC.projects.DeltaSDC.mlsdc import round_value

    with pytest.raises(NotImplementedError, match='cannot be rounded'):
        round_value(object(), np.dtype('float32'))


def iterations(multilevel, eta_fine=None, eta_coarse=None, maxiter=40):
    """Iterations to reach 1e-11 with node-local solves that deliver only `eta`."""
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
    from pySDC.projects.DeltaSDC.mlsdc import delta_implicit_rounded, delta_transfer
    from pySDC.projects.DeltaSDC.tests.inexact_problem import heat_inexact

    nvars = [127, 63] if multilevel else [127]
    description = {
        'problem_class': heat_inexact,
        'problem_params': dict(HEAT_PARAMS, nvars=nvars, eta=[eta_fine, eta_coarse] if multilevel else [eta_fine]),
        'sweeper_class': delta_implicit_rounded,
        'sweeper_params': sweeper_params(),
        'level_params': {'restol': -1, 'dt': 1e-1},
        'step_params': {'maxiter': maxiter},
        'space_transfer_class': mesh_to_mesh,
        'space_transfer_params': {'iorder': 4, 'rorder': 2},
        'base_transfer_class': delta_transfer,
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    prob = controller.MS[0].levels[0].prob
    _, stats = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=1e-1)
    residuals = [value for _, value in get_sorted(stats, type='residual_post_iteration', sortby='iter')]
    return next((i + 1 for i, value in enumerate(residuals) if value < 1e-11), None)


@pytest.mark.base
def test_delivered_accuracy_requirement():
    """
    What the node-local solver has to *deliver*, which is all the iteration can see.

    The point of measuring it this way rather than as a format is that it says nothing about how a
    solver reaches the accuracy -- low-precision kernels with iterative refinement, a loose Krylov
    tolerance, a few multigrid cycles are all the same thing here. These are the numbers the README
    quotes, and the third one is the control: the coarse solve tolerates four orders of magnitude
    more error than the fine solve in the same run, which is the whole claim.
    """
    sdc, mlsdc = iterations(False), iterations(True)
    assert (sdc, mlsdc) == (14, 7), f'baselines moved, got {sdc} and {mlsdc}'

    # SDC's fine solve: about four digits
    assert iterations(False, eta_fine=1e-4) <= sdc + 1
    assert iterations(False, eta_fine=1e-2) > sdc + 1

    # MLSDC's fine solve: about six digits, so two orders tighter than SDC's
    assert iterations(True, eta_fine=1e-6) <= mlsdc + 1
    assert iterations(True, eta_fine=1e-4) > mlsdc + 1

    # MLSDC's coarse solve: about two digits, four orders looser than its own fine solve
    assert iterations(True, eta_coarse=1e-2) <= mlsdc + 1
    assert iterations(True, eta_coarse=1e-1) > mlsdc + 1


def converge(prob_extra=None, **sweep_extra):
    """Run two-level heat to a residual tolerance and return the iterations and the end value."""
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
    from pySDC.projects.DeltaSDC.mlsdc import delta_implicit_rounded, delta_transfer
    from pySDC.projects.DeltaSDC.problems import heat_delta

    description = {
        'problem_class': heat_delta,
        'problem_params': dict(HEAT_PARAMS, nvars=[127, 63], **(prob_extra or {})),
        'sweeper_class': delta_implicit_rounded,
        'sweeper_params': sweeper_params(**sweep_extra),
        'level_params': {'restol': 1e-11, 'dt': 1e-1},
        'step_params': {'maxiter': 40},
        'space_transfer_class': mesh_to_mesh,
        'space_transfer_params': {'iorder': 4, 'rorder': 2},
        'base_transfer_class': delta_transfer,
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    prob = controller.MS[0].levels[0].prob
    uend, stats = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=1e-1)
    return len(get_sorted(stats, type='residual_post_iteration', sortby='iter')), uend


@pytest.mark.base
@pytest.mark.parametrize('precision', ['float32', 'float16'])
def test_emulation_agrees_with_the_real_thing(precision):
    """
    The emulation has to predict what a genuine dtype does, or the measurements made with it do not
    transfer.

    Both knobs are needed to model it: a level that genuinely stores at ``float16`` also holds its
    *corrections* there, which ``level_precision`` alone does not capture -- and on this
    configuration ``level_precision`` alone is entirely vacuous, because a linear operator reached
    through ``linear_implicit`` never reads the coarse state at all.
    """
    genuine_count, genuine = converge({'dtype': ['float64', precision]})
    emulated_count, emulated = converge(
        level_precision=[None, np.dtype(precision)], correction_precision=[None, np.dtype(precision)]
    )

    # Within one iteration rather than exactly equal: the two do the same arithmetic in a different
    # order, and under NEP 50 -- NumPy 2's promotion rule, reachable here with
    # NPY_PROMOTION_STATE=weak -- that is worth one iteration on the genuine side. A tolerance of one
    # still separates them from `level_precision` alone, which is at least three iterations away.
    assert abs(genuine_count - emulated_count) <= 1, f'{genuine_count} genuine, {emulated_count} emulated'
    assert abs(genuine - emulated) < 1e-11, f'the two differ by {abs(genuine - emulated):.3e}'


@pytest.mark.base
def test_level_precision_alone_understates_float16():
    """
    The control for the test above: the two knobs are not interchangeable.

    ``level_precision`` alone reports half precision on the coarse level as free, and a genuine
    ``float16`` level costs four iterations under NumPy 2's promotion rule, three under NumPy 1's --
    hence the assertion on "more than one", which holds either way. Without this the agreement above
    could be read as "the emulation was right all along", which is only true of the two knobs
    together.
    """
    baseline, _ = converge()
    genuine, _ = converge({'dtype': ['float64', 'float16']})
    level_only, _ = converge(level_precision=[None, np.float16])

    assert level_only == baseline, 'level_precision alone is supposed to look free here'
    assert genuine > baseline + 1, f'genuine float16 came out free ({genuine} vs {baseline})'


@pytest.mark.base
@pytest.mark.parametrize('precision', [np.float32, np.float16])
def test_emulated_level_precision_really_rounds(precision):
    """
    The emulation keeps ``float64`` containers on purpose, so the dtype proves nothing there.

    What has to hold instead is that the values are exactly representable in the requested format --
    and that the fine level's are not, which is what says the rounding is being applied where it was
    asked for and nowhere else.
    """
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
    from pySDC.projects.DeltaSDC.mlsdc import delta_implicit_rounded, delta_transfer
    from pySDC.projects.DeltaSDC.problems import heat_delta

    description = {
        'problem_class': heat_delta,
        'problem_params': dict(HEAT_PARAMS, nvars=[127, 63]),
        'sweeper_class': delta_implicit_rounded,
        'sweeper_params': dict(SWEEPER_PARAMS, level_precision=[None, precision]),
        'level_params': {'restol': 1e-11, 'dt': 1e-1},
        'step_params': {'maxiter': 40},
        'space_transfer_class': mesh_to_mesh,
        'space_transfer_params': {'iorder': 4, 'rorder': 2},
        'base_transfer_class': delta_transfer,
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    prob = controller.MS[0].levels[0].prob
    controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=1e-1)

    def values(level):
        return [np.asarray(v) for name in ('u', 'f') for v in getattr(level, name) if v is not None]

    fine, coarse = controller.MS[0].levels
    assert all(np.array_equal(v, v.astype(precision)) for v in values(coarse)), 'the coarse level was not rounded'
    assert not all(np.array_equal(v, v.astype(precision)) for v in values(fine)), 'the fine level was rounded too'
