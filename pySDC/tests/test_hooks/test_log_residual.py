import pytest


def run(monkeypatch, num_procs, num_levels, hook_class=None):
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh

    # patched on the class, not in a subclass here: the steps are copies made with dill, which would copy a local
    # class and the list it appends to
    calls = []
    compute_residual = generic_implicit.compute_residual

    def counted(self, stage=''):
        calls.append(stage)
        return compute_residual(self, stage=stage)

    monkeypatch.setattr(generic_implicit, 'compute_residual', counted)

    description = {
        'problem_class': heatNd_unforced,
        'problem_params': {'nvars': [64, 32][:num_levels], 'nu': 0.1, 'freq': 2, 'bc': 'periodic'},
        'sweeper_class': generic_implicit,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': [3, 2][:num_levels], 'QI': 'LU'},
        'level_params': {'dt': 0.05, 'restol': 1e-9},
        'step_params': {'maxiter': 30},
        'space_transfer_class': mesh_to_mesh,
        'space_transfer_params': {'iorder': 6, 'rorder': 2, 'periodic': True},
    }
    controller_params = {'logger_level': 30, 'hook_class': hook_class or []}
    controller = controller_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)
    prob = controller.MS[0].levels[0].prob
    _, stats = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=8 * 0.05)
    return stats, calls


@pytest.mark.base
@pytest.mark.parametrize('num_procs, num_levels', [(1, 1), (1, 2), (4, 2), (4, 1)])
def test_one_residual_per_iteration(monkeypatch, num_procs, num_levels):
    """
    The controller computes the residual once per iteration, where the convergence check reads it, plus once after
    the predictor -- not after every sweep on every level as well.
    """
    from pySDC.helpers.stats_helper import get_sorted

    stats, calls = run(monkeypatch, num_procs, num_levels)
    niter = [v for _, v in get_sorted(stats, type='niter')]
    assert set(calls) == {'IT_CHECK'}, f'residuals computed outside of IT_CHECK: {set(calls)}'
    assert len(calls) == sum(niter) + len(niter), f'{len(calls)} residuals for {sum(niter)} iterations'


@pytest.mark.base
@pytest.mark.parametrize('num_procs, num_levels', [(1, 1), (4, 2)])
def test_log_residual_post_sweep(monkeypatch, num_procs, num_levels):
    """
    The hook records the residual after every sweep on every level, and asking for it changes nothing.
    """
    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.hooks.log_residual import LogResidualPostSweep

    plain, _ = run(monkeypatch, num_procs, num_levels)
    hooked, calls = run(monkeypatch, num_procs, num_levels, hook_class=[LogResidualPostSweep])

    for kind in ['niter', 'residual_post_iteration']:
        assert get_sorted(plain, type=kind) == get_sorted(hooked, type=kind), f'the hook changed {kind}'

    for level in range(num_levels):
        post_sweep = get_sorted(hooked, type='residual_post_sweep', level=level)
        assert len(post_sweep) == calls.count('IT_FINE' if level == 0 else 'IT_COARSE') > 0
        assert all(np.isfinite(v) for _, v in post_sweep)

    if num_procs == 1 and num_levels == 1:
        # nothing comes in between the sweep and the convergence check, so both see the same residual
        post_iter = [v for _, v in get_sorted(hooked, type='residual_post_iteration', sortby='iter')]
        post_sweep = [v for _, v in get_sorted(hooked, type='residual_post_sweep', sortby='iter')]
        assert post_iter == post_sweep
