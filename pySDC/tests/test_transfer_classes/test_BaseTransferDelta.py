r"""
Tests for :mod:`pySDC.implementations.transfer_classes.BaseTransferDelta`.

The hierarchy replaces two cancellations of :math:`\mathcal{O}(1)` quantities with exact
identities. The first three tests check the identities; the rest check that using them changes no
answer, at two, three and four levels and under PFASST.

What the rewrite *buys* -- a coarse level below backend precision -- needs controls that must break,
and is measured separately. What is asserted here is the other half: that using the identities costs
nothing at backend precision.
"""

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


def build(nvars, sweeper_class, transfer_class, num_procs=1, maxiter=6, dt=1e-1, **extra):
    """Build a controller on the heat equation with the given hierarchy."""
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced

    description = {
        'problem_class': heatNd_unforced,
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


def tau_building(transfer_class):
    """
    A transfer that builds the FAS tau even though the delta hierarchy does not read it.

    The two identity tests below compare the inherited residual against the one a coarse level would
    have rebuilt from its own state, and that rebuild needs tau. The lean path deliberately skips
    building it, so the comparison is made on the path where the quantity exists;
    :func:`test_lean_restriction_changes_nothing` then checks that skipping it changes no answer.
    """

    class with_tau(transfer_class):
        def coarse_reads_tau(self):
            return True

    return with_tau


def nrm(value):
    import numpy as np

    return float(np.max(np.abs(np.asarray(value))))


@pytest.mark.base
def test_coarse_residual_is_the_restricted_fine_residual():
    """
    The identity the coarse level's residual is replaced by.

    Stock MLSDC rebuilds ``eps_G = u_G[0] + tau + dt (Q f_G) - u_G[m]`` from O(1) coarse data;
    analytically that *is* the restricted fine residual, which is what the FAS tau is for. If this
    identity did not hold, the hierarchy would be solving a different problem.
    """
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit
    from pySDC.implementations.transfer_classes.BaseTransferDelta import delta_transfer

    deviations = []

    class checking_transfer(tau_building(delta_transfer)):
        def restrict(self):
            super().restrict()
            # what the coarse level would have computed the stock way, from its own O(1) state
            rebuilt = delta_implicit._residual_nodes(self.coarse.sweep)
            inherited = self.coarse.sweep.eps_in
            deviations.append(max(nrm(a - b) for a, b in zip(rebuilt, inherited)))

    run([127, 63], delta_implicit, checking_transfer)

    assert len(deviations) > 1, 'the transfer never ran'
    assert max(deviations) < 1e-13, f'the two residuals differ by {max(deviations):.3e}'


@pytest.mark.base
def test_accumulated_correction_is_the_coarse_grid_correction():
    """
    The identity the coarse-grid correction is replaced by.

    Stock MLSDC recovers it as ``G.u - G.uold``, a difference of two O(1) states; it is also the sum
    of the sweep's own increments, which is what the delta transfer prolongs instead.
    """
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit
    from pySDC.implementations.transfer_classes.BaseTransferDelta import delta_transfer

    deviations = []

    class checking_transfer(delta_transfer):
        def prolong(self):
            accumulated = self.coarse.sweep.delta_acc
            differenced = [self.coarse.u[m + 1] - self.coarse.uold[m + 1] for m in range(len(accumulated))]
            deviations.append(max(nrm(a - b) for a, b in zip(accumulated, differenced)))
            super().prolong()

    run([127, 63], delta_implicit, checking_transfer)

    assert len(deviations) > 1, 'the transfer never ran'
    assert max(deviations) < 1e-13, f'the two corrections differ by {max(deviations):.3e}'


@pytest.mark.base
def test_residual_recursion_tracks_the_real_residual():
    """
    ``eps <- eps - delta + dt (Q df)`` must keep agreeing with the residual recomputed from scratch.

    This is what lets a level carry its residual through sweeps and prolongations instead of
    rebuilding it out of O(1) state, so it has to hold after every one of them.
    """
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit
    from pySDC.implementations.transfer_classes.BaseTransferDelta import delta_transfer

    deviations = []

    class checking_sweeper(delta_implicit):
        def update_nodes(self):
            super().update_nodes()
            if self.eps_in is not None:
                rebuilt = delta_implicit._residual_nodes(self)
                deviations.append(max(nrm(a - b) for a, b in zip(rebuilt, self.eps_in)))

    # three levels, so a middle level is swept, restricted from, prolonged onto and swept again
    run([127, 63, 31], checking_sweeper, tau_building(delta_transfer))

    assert len(deviations) > 1, 'the recursion never ran'
    assert max(deviations) < 1e-13, f'the tracked residual drifted by {max(deviations):.3e}'


@pytest.mark.base
@pytest.mark.parametrize('nvars', [[127, 63], [127, 63, 31], [127, 63, 31, 15]])
def test_matches_stock_mlsdc(nvars):
    """At backend precision the delta hierarchy is an algebraic rewrite, so it must change nothing."""
    from pySDC.core.base_transfer import BaseTransfer
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit
    from pySDC.implementations.transfer_classes.BaseTransferDelta import delta_transfer

    stock = run(nvars, delta_implicit, BaseTransfer)
    delta = run(nvars, delta_implicit, delta_transfer)
    assert abs(stock - delta) < 1e-13, f'{len(nvars)}-level MLSDC deviates by {abs(stock - delta):.3e}'


@pytest.mark.base
@pytest.mark.parametrize('num_procs', [2, 4])
def test_pfasst(num_procs):
    """Multiple levels and multiple steps in a block, which MLSDC alone does not cover."""
    from pySDC.core.base_transfer import BaseTransfer
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit
    from pySDC.implementations.transfer_classes.BaseTransferDelta import delta_transfer

    stock = run([127, 63], delta_implicit, BaseTransfer, num_procs=num_procs, nsteps=num_procs)
    delta = run([127, 63], delta_implicit, delta_transfer, num_procs=num_procs, nsteps=num_procs)
    assert abs(stock - delta) < 1e-13, f'PFASST on {num_procs} steps deviates by {abs(stock - delta):.3e}'


@pytest.mark.base
@pytest.mark.parametrize('nvars', [[127, 63], [127, 63, 31]])
def test_lean_restriction_changes_nothing(nvars):
    """
    Skipping the tau construction is an optimisation, so it has to be invisible in the answer.

    The delta hierarchy reads the restricted fine residual instead, and its own
    ``compute_residual`` reports the residual it already tracks, so nothing is left that reads tau --
    except ``compute_end_point`` in the quadrature-update case, which is what the gate is for.
    """
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit
    from pySDC.implementations.transfer_classes.BaseTransferDelta import delta_transfer

    lean = run(nvars, delta_implicit, delta_transfer)
    built = run(nvars, delta_implicit, tau_building(delta_transfer))
    assert abs(lean - built) < 1e-13, f'the lean restriction moved the answer by {abs(lean - built):.3e}'


@pytest.mark.base
def test_coarse_level_reports_the_residual_it_tracks():
    """
    A coarse level's reported residual must still be the FAS residual, not a rebuild without tau.

    It is only ever logged, so a wrong value here would not fail anything else -- which is exactly
    why it is checked. Compared against what the stock hierarchy reports for the same level.
    """
    from pySDC.core.base_transfer import BaseTransfer
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit
    from pySDC.implementations.transfer_classes.BaseTransferDelta import delta_transfer

    seen = {}

    def watched(transfer_class, key):
        class watcher(transfer_class):
            def prolong(self):
                seen.setdefault(key, []).append(self.coarse.status.residual)
                super().prolong()

        return watcher

    stock = build([127, 63], delta_implicit, watched(BaseTransfer, 'stock'))
    delta = build([127, 63], delta_implicit, watched(delta_transfer, 'delta'))
    for controller in (stock, delta):
        prob = controller.MS[0].levels[0].prob
        controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=1e-1)

    assert len(seen['stock']) > 2, 'the coarse level never reported a residual'
    assert len(seen['stock']) == len(seen['delta'])
    for a, b in zip(seen['stock'], seen['delta']):
        # The two routes differ by round-off, which stays around 1e-14 absolute while the residual
        # falls, so the bar has to be absolute as well as relative. It still discriminates: a
        # residual rebuilt without tau differs by O(tau), i.e. by the coarsening defect.
        assert abs(a - b) < 1e-9 * abs(a) + 1e-13, f'coarse residual differs: {a:.3e} vs {b:.3e}'


@pytest.mark.base
def test_lean_restriction_is_actually_taken():
    """
    The saving is only real if the gate lets it through, so check that tau is never built.

    With a right-sided rule and no collocation update -- pySDC's default, and what every example
    here uses -- nothing reads tau, and a coarse level that still carries one means the gate has
    silently stopped opening.
    """
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit
    from pySDC.implementations.transfer_classes.BaseTransferDelta import delta_transfer

    controller = build([127, 63], delta_implicit, delta_transfer)
    prob = controller.MS[0].levels[0].prob
    controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=1e-1)

    coarse = controller.MS[0].levels[1]
    assert coarse.status.unlocked, 'the coarse level never ran'
    assert all(t is None for t in coarse.tau), 'tau was built even though nothing reads it'

    # and the gate closes again when the end point comes from the quadrature update, which is the
    # one place tau is still read
    with_update = build([127, 63], delta_implicit, delta_transfer, do_coll_update=True)
    prob = with_update.MS[0].levels[0].prob
    with_update.run(u0=prob.u_exact(0.0), t0=0.0, Tend=1e-1)
    assert any(t is not None for t in with_update.MS[0].levels[1].tau), 'tau is needed here and was skipped'


@pytest.mark.base
def test_tau_is_substituted_not_discarded():
    r"""
    The control for :func:`test_matches_stock_mlsdc`, and the one that says what is going on.

    This hierarchy never builds the FAS :math:`\tau`, which reads like MLSDC's defining term being
    thrown away. It is not: :math:`\tau = R(\Delta t\,Q_F f_F) - \Delta t\,Q_G f_G` put into the
    coarse residual cancels the :math:`\Delta t\,Q_G f_G` terms identically and leaves
    :math:`R\varepsilon_F`, so a level handed that residual is solving the FAS-corrected problem and
    would double-count if it added :math:`\tau` as well.

    Agreeing with stock MLSDC proves that only if :math:`\tau` matters on this configuration, so the
    third row throws it away for real. It does not converge at all.
    """
    import numpy as np
    from pySDC.core.base_transfer import BaseTransfer
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit
    from pySDC.implementations.transfer_classes.BaseTransferDelta import delta_transfer

    sizes = []

    class measuring(BaseTransfer):
        def restrict(self):
            super().restrict()
            sizes.append(max(nrm(t) for t in self.coarse.tau))

    class tau_zeroed(BaseTransfer):
        """Stock MLSDC with the FAS correction discarded rather than substituted."""

        def restrict(self):
            super().restrict()
            for m in range(len(self.coarse.tau)):
                self.coarse.tau[m] *= 0.0

    stock = floor([127, 63], delta_implicit, measuring, maxiter=25)
    delta = floor([127, 63], delta_implicit, delta_transfer, maxiter=25)
    zeroed = floor([127, 63], delta_implicit, tau_zeroed, maxiter=25)

    reference = run([127, 63], delta_implicit, BaseTransfer, maxiter=25)
    rewritten = run([127, 63], delta_implicit, delta_transfer, maxiter=25)

    # tau is a real correction here, not a rounding-level one
    assert max(sizes) > 1e-4, f'tau is only {max(sizes):.2e}, so discarding it would prove nothing'
    assert stock < 1e-12 and delta < 1e-12, f'the converging rows did not converge: {stock:.2e}, {delta:.2e}'
    assert abs(rewritten - reference) < 1e-13, f'the rewrite moved the answer by {abs(rewritten - reference):.3e}'
    assert zeroed > 1e-8, f'discarding tau floored at {zeroed:.2e}, so this controls nothing'
    assert not np.isclose(zeroed, delta), 'discarding tau and substituting it cannot be the same thing'
