r"""
Tests for levels that genuinely run at a reduced dtype.

``dtype`` on the finite-difference problems gives a level whose arrays really are ``float32`` or
``float16``. That is easy to get wrong in a way that looks right: a silent upcast agrees with
``float64`` perfectly.

So these assert in both directions. The reduced dtype has to survive a whole run rather than drift
back to ``float64``; and it has to *change* the numbers by roughly the format's epsilon, which is
what a run that never left ``float64`` could not do.

The sweeper's own reduced-precision storage, ``correction_precision``, is a different mechanism and
is tested with the sweeper in ``pySDC/tests/test_sweepers/test_delta_form.py``.
"""

import numpy as np
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


def run(prob_extra=None, **sweep_extra):
    """Run one step of two-level heat and return the iterations, the end value and the controller."""
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit
    from pySDC.implementations.transfer_classes.BaseTransferDelta import delta_transfer

    description = {
        'problem_class': heatNd_unforced,
        'problem_params': dict(HEAT_PARAMS, nvars=[127, 63], **(prob_extra or {})),
        'sweeper_class': delta_implicit,
        'sweeper_params': dict(SWEEPER_PARAMS, **sweep_extra),
        'level_params': {'restol': 1e-11, 'dt': 1e-1},
        'step_params': {'maxiter': 40},
        'space_transfer_class': mesh_to_mesh,
        'space_transfer_params': {'iorder': 4, 'rorder': 2},
        'base_transfer_class': delta_transfer,
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    prob = controller.MS[0].levels[0].prob
    uend, stats = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=1e-1)
    residuals = [value for _, value in get_sorted(stats, type='residual_post_iteration', sortby='iter')]
    return len(residuals), uend, controller


@pytest.mark.base
@pytest.mark.parametrize('precision', ['float32', 'float16'])
def test_the_reduced_dtype_survives_the_run(precision):
    """
    Every array the coarse level holds must still be at the requested precision when the run ends.

    This is the check the emulation cannot make, and the one that would catch a silent upcast. A
    coefficient taken out of a numpy array is an ``np.float64``, and under NEP 50 multiplying a
    reduced-precision array by one produces ``float64`` -- so a single missing ``float()`` puts the
    level back at backend precision without changing an answer or failing anything else.
    """
    _, _, controller = run({'dtype': ['float64', precision]})
    fine, coarse = controller.MS[0].levels

    held = {str(np.asarray(v).dtype) for v in list(coarse.u) + list(coarse.f) if v is not None}
    assert held == {precision}, f'the coarse level drifted to {held}'
    assert {str(np.asarray(v).dtype) for v in fine.u if v is not None} == {'float64'}
    assert coarse.prob.A.dtype == np.promote_types(precision, np.float32)


@pytest.mark.base
def test_reduced_storage_actually_costs_fewer_bytes():
    """The point of a real dtype over the emulation is the memory, so check it is actually smaller."""
    sizes = {}
    for precision in ('float64', 'float32', 'float16'):
        _, _, controller = run({'dtype': ['float64', precision]})
        coarse = controller.MS[0].levels[1]
        sizes[precision] = sum(np.asarray(v).nbytes for v in list(coarse.u) + list(coarse.f) if v is not None)

    assert sizes['float32'] == sizes['float64'] // 2
    assert sizes['float16'] == sizes['float64'] // 4


@pytest.mark.base
def test_default_dtype_is_unchanged():
    """Adding the parameter must not have moved anything for callers who never pass it."""
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced

    prob = heatNd_unforced(nvars=127, **HEAT_PARAMS)
    assert prob.init[-1] == np.dtype('float64')
    assert prob.A.dtype == np.dtype('float64')
    assert prob.u_exact(0.0).dtype == np.dtype('float64')


@pytest.mark.base
def test_float16_state_still_does_not_work():
    """
    The control, and the one thing no reformulation moves: the fine level's state stays float64.

    That is where the residual is formed by cancelling O(1) quantities, so there is nothing to scale
    -- the level is the state. Half precision there barely converges at all.
    """
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit

    floors = {}
    for precision in ('float64', 'float32', 'float16'):
        description = {
            'problem_class': heatNd_unforced,
            'problem_params': dict(HEAT_PARAMS, nvars=127, dtype=precision),
            'sweeper_class': delta_implicit,
            'sweeper_params': dict(SWEEPER_PARAMS),
            'level_params': {'restol': -1, 'dt': 1e-1},
            'step_params': {'maxiter': 30},
        }
        controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
        prob = controller.MS[0].levels[0].prob
        _, stats = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=1e-1)
        floors[precision] = min(v for _, v in get_sorted(stats, type='residual_post_iteration', sortby='iter'))

    assert floors['float64'] < 1e-12
    assert floors['float32'] > 1e-6, 'a float32 state stopped capping the fine level'
    assert floors['float16'] > 1e-2, 'a float16 state stopped capping the fine level'


@pytest.mark.base
@pytest.mark.parametrize('precision,epsilon', [('float32', 1.2e-7), ('float16', 9.8e-4)])
def test_reduced_precision_leaves_its_fingerprint(precision, epsilon):
    """
    A reduced level must *change* the numbers, by roughly the format's epsilon.

    Every other test here asserts that reduced precision agrees with ``float64`` -- which is also
    exactly what would be seen if a bug silently upcast everything and no reduction happened at all.
    This asserts the opposite: the coarse level's state after its first sweep has to differ from the
    ``float64`` run by something between round-off and the format's own epsilon. Too small and the
    reduction is not happening; too large and it is not the reduction doing it.
    """
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit
    from pySDC.implementations.transfer_classes.BaseTransferDelta import delta_transfer

    captured = {}

    class capture(delta_implicit):
        def update_nodes(self):
            super().update_nodes()
            if self.level.level_index == 1 and 'state' not in captured:
                captured['state'] = np.asarray(self.level.u[1], dtype=np.float64).copy()

    def snapshot(coarse_dtype):
        captured.clear()
        description = {
            'problem_class': heatNd_unforced,
            'problem_params': dict(HEAT_PARAMS, nvars=[127, 63], dtype=['float64', coarse_dtype]),
            'sweeper_class': capture,
            'sweeper_params': dict(SWEEPER_PARAMS),
            'level_params': {'restol': 1e-11, 'dt': 1e-1},
            'step_params': {'maxiter': 40},
            'space_transfer_class': mesh_to_mesh,
            'space_transfer_params': {'iorder': 4, 'rorder': 2},
            'base_transfer_class': delta_transfer,
        }
        controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
        prob = controller.MS[0].levels[0].prob
        controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=1e-1)
        return captured['state']

    reference = snapshot('float64')
    # the control: the harness is deterministic, so float64 against itself has to be exactly zero,
    # or the differences below would mean nothing
    assert np.array_equal(snapshot('float64'), reference)

    relative = np.max(np.abs(snapshot(precision) - reference)) / np.max(np.abs(reference))
    assert relative > epsilon / 100, f'{precision} changed nothing ({relative:.1e}) -- is it really reduced?'
    assert relative < epsilon * 100, f'{precision} changed too much ({relative:.1e}) to be rounding'


@pytest.mark.base
@pytest.mark.parametrize('precision', ['float32', 'float16'])
def test_the_transfer_operators_follow_the_level(precision):
    """
    The space transfer has to carry its operators at the precision of what they produce.

    A ``float64`` operator applied to a reduced-precision vector upcasts, so the transfer would do
    its work in double and only round on the way into the destination -- the one place a
    reduced-precision level would keep paying full freight while every dtype assertion still passed.
    So this checks the operators *and* that what comes out of a restriction is not wider than the
    level it lands on.
    """
    _, _, controller = run({'dtype': ['float64', precision]})
    transfer = controller.MS[0].base_transfer.space_transfer
    expected = np.promote_types(precision, np.float32)

    assert transfer.Rspace.dtype == expected, f'the restriction operator is {transfer.Rspace.dtype}'
    assert transfer.Pspace.dtype == np.dtype('float64'), 'the prolongation produces a float64 level'

    fine, coarse = controller.MS[0].levels
    restricted = transfer.restrict(fine.u[1])
    assert np.asarray(restricted).dtype == np.dtype(
        precision
    ), f'a restriction onto a {precision} level produced {np.asarray(restricted).dtype}'
    assert np.asarray(transfer.prolong(coarse.u[1])).dtype == np.dtype('float64')


@pytest.mark.base
def test_transfer_operators_are_unchanged_by_default():
    """Adding the cast must not have moved anything for callers who never pass a dtype."""
    _, _, controller = run()
    transfer = controller.MS[0].base_transfer.space_transfer

    assert transfer.Rspace.dtype == np.dtype('float64')
    assert transfer.Pspace.dtype == np.dtype('float64')
