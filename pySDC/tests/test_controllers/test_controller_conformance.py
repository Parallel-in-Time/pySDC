"""
Conformance suite: the two transports of one algorithm have to be the same program.

pySDC implements each time-parallel algorithm twice -- once with every step in one process
(``controller_nonMPI``) and once with one step per rank (``controller_MPI``). Running virtually
rather than in parallel is meant to be a *decomposition parameter*, not a different program, so a
user who swaps the controller must get the same answer and the same features.

Today they are not the same program, and nothing in the test suite says so. This file is where that
gets asserted, one test per axis on which they can drift.

What this file deliberately does NOT duplicate:

- ParaDiag's cross-transport equivalence, which ``test_controller_ParaDiag_MPI.py`` already covers
  in ParaDiag's own terms (``test_ParaDiag_MPI_matches_nonMPI`` and friends, at ``rtol=1e-14``).
  The axes here are the ones that apply to *any* controller pair.
- Algorithm correctness. Whether SDC converges, or ParaDiag has the right order, is tested
  elsewhere; conformance only asks whether the two transports agree with *each other*.
- ``test_controller_equivalence.py`` on the ``feat/timecomm-virtual`` branch, which was the Phase 0
  characterisation test. ``test_pfasst_baseline`` here supersedes it -- delete that file when the
  branch is rebased rather than keeping both.
- ``AdaptiveAlpha``'s cross-transport agreement, pinned by ``test_adaptive_alpha.py`` and tutorial
  step_9 E. The block-hook axis here checks the *contract*; those check one user of it.

Two of the axes here are not about the transports agreeing but about what they must both keep
doing: an early step has to be able to converge and drop out, and ``all_to_done`` has to keep
turning that off on request. Both controllers can agree with each other perfectly while having lost
the pipelining, so agreement alone is not enough to pin.

Axes marked ``xfail`` are known asymmetries with a documented cause. They are strict, so fixing one
without removing its marker fails the suite: the marker is a to-do list, not a suppression.

A note on the iteration estimator, since that axis asserts something weaker than parity.
``controller_MPI`` used to implement it inline while ``controller_nonMPI`` did not implement it at
all, and the supported route was a third thing -- the ``CheckIterationEstimatorNonMPI`` convergence
controller, which tutorial step_8 C exercises. Only that third one ever worked, so the inline
implementation was removed and both controllers now refuse the parameter identically. The
convergence controller remains nonMPI-only, so the *feature* is still not symmetric; what the axis
pins is that the two controllers no longer disagree in silence.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from pySDC.core.convergence_controller import ConvergenceController

ATOL = 1e-14
RTOL = 0.0
DT = 0.1
NUM_PROCS = 4


class RecursionProbe(ConvergenceController):
    """
    A convergence controller that carries iteration-indexed state on ``self``, and records it.

    This is the smallest thing that can detect the call-count contract: a convergence controller is
    instantiated once per *controller*, so ``self`` spans a whole block virtually but a single rank
    under MPI, while ``post_iteration_processing`` fires once per *step*. Anything recursive on
    ``self`` therefore advances a different number of times in the two transports.

    Deliberately not a real convergence controller -- it changes nothing about the run, so it can be
    added to any configuration without perturbing the numbers the other axes compare.
    """

    def setup(self, controller, params, description, **kwargs):
        return {'control_order': +500, **super().setup(controller, params, description, **kwargs)}

    def setup_status_variables(self, controller, **kwargs):
        self.calls = []
        self.block_calls = 0
        return None

    def post_iteration_processing(self, controller, S, **kwargs):
        self.calls.append((int(S.status.slot), int(S.status.iter)))
        return None

    def post_iteration_processing_block(self, controller, **kwargs):
        self.block_calls += 1
        return None


def get_description(config):
    """Single-level IMEX SDC, or the same problem as two-level MLSDC/PFASST."""
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced
    from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
    from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh

    description = {
        'problem_class': heatNd_forced,
        'problem_params': {'nu': 0.1, 'freq': (2,), 'nvars': (15,), 'bc': 'dirichlet-zero'},
        'sweeper_class': imex_1st_order,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'IE'},
        'level_params': {'restol': 1e-09, 'dt': DT},
        'step_params': {'maxiter': 10},
    }
    if config == 'multi_level':
        description['problem_params']['nvars'] = [(31,), (15,)]
        description['sweeper_params']['num_nodes'] = [3, 2]
        description['space_transfer_class'] = mesh_to_mesh
        description['space_transfer_params'] = {'rorder': 2, 'iorder': 6}
    if config == 'gauss_seidel':
        # tuned so the steps converge one after another and the running set drains 4, 3, 2, 1 --
        # see ``test_gauss_seidel_mssdc_agrees`` for why reaching one matters
        description['sweeper_params']['num_nodes'] = 2
        description['level_params'] = {'restol': 1e-10, 'dt': 0.8}
        description['step_params'] = {'maxiter': 99}
    return description


def run(useMPI, config='single_level', probe=False, all_to_done=False, mssdc_jac=True, num_procs=NUM_PROCS):
    """
    Run one configuration through one of the two transports and return everything the axes compare.

    Returns:
        dict: uend, the (time, niter) pairs, the sorted stats types, and the probe's final state
    """
    from pySDC.helpers.stats_helper import get_sorted

    description = get_description(config)
    controller_params = {'logger_level': 30, 'all_to_done': all_to_done, 'mssdc_jac': mssdc_jac}
    if probe:
        description['convergence_controllers'] = {RecursionProbe: {}}

    if useMPI:
        from mpi4py import MPI
        from pySDC.implementations.controller_classes.controller_MPI import controller_MPI

        comm = MPI.COMM_WORLD
        controller = controller_MPI(comm=comm, controller_params=controller_params, description=description)
        P = controller.S.levels[0].prob
    else:
        from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

        comm = None
        controller = controller_nonMPI(
            num_procs=num_procs, controller_params=controller_params, description=description
        )
        P = controller.MS[0].levels[0].prob

    dt = description['level_params']['dt']
    uend, stats = controller.run(u0=P.u_exact(0.0), t0=0.0, Tend=num_procs * dt)

    probes = [me for me in controller.convergence_controllers if type(me) == RecursionProbe]
    return {
        'uend': np.asarray(uend),
        'niter': np.array(get_sorted(stats, type='niter', sortby='time', comm=comm), dtype=float),
        'stats_types': sorted({me.type for me in stats.keys()}),
        'block_calls': probes[0].block_calls if probes else -1,
        'iters_seen': len({it for _, it in probes[0].calls}) if probes else -1,
    }


def iteration_estimator_rejected(useMPI, num_procs=NUM_PROCS):
    """
    Ask a controller to build with ``use_iteration_estimator`` and report whether it refused.

    Returns:
        bool: True if the controller raised rather than accepting the parameter
    """
    from pySDC.core.errors import ControllerError

    description = get_description('single_level')
    controller_params = {'logger_level': 30, 'use_iteration_estimator': True}

    try:
        if useMPI:
            from mpi4py import MPI
            from pySDC.implementations.controller_classes.controller_MPI import controller_MPI

            controller_MPI(comm=MPI.COMM_WORLD, controller_params=controller_params, description=description)
        else:
            from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

            controller_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)
    except ControllerError:
        return True
    return False


CASES = {
    'baseline_single': {'config': 'single_level'},
    'baseline_multi': {'config': 'multi_level'},
    'probe': {'config': 'single_level', 'probe': True},
    'global_convergence': {'config': 'single_level', 'all_to_done': True},
    'gauss_seidel': {'config': 'gauss_seidel', 'mssdc_jac': False},
}


@pytest.fixture(scope='module')
def results(tmp_path_factory):
    """
    Run every case through both transports, launching ``mpirun`` exactly once.

    Starting an MPI job costs about a minute here and dominates everything this file does -- the
    solves themselves are milliseconds. One launch that covers every case therefore costs what a
    single test used to, which is the difference between a suite worth having in CI and one that
    gets deleted the next time somebody trims the pipeline (cf. #675).
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    out = str(tmp_path_factory.mktemp('conformance') / 'mpi_results.npz')

    env = os.environ.copy()
    env['PYTHONPATH'] = root
    env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

    cmd = f'mpirun -np {NUM_PROCS} {sys.executable} {__file__} {out}'
    p = subprocess.Popen(cmd.split(), env=env, cwd=root)
    p.wait()
    assert p.returncode == 0, f'ERROR: mpirun returned {p.returncode}'

    loaded = np.load(out, allow_pickle=True)
    out_dict = {}
    for name, kwargs in CASES.items():
        mpi = {k.split('/', 1)[1]: loaded[k] for k in loaded.files if k.startswith(f'{name}/')}
        out_dict[name] = (run(useMPI=False, **kwargs), mpi)
    out_dict['estimator_rejected_mpi'] = bool(loaded['estimator/rejected'])
    return out_dict


@pytest.mark.base
def test_multilevel_visits_all_stages():
    """
    Without this, a green baseline could still be covering only 4 of the 7 stages.

    ``it_check`` only sends a step to ``IT_DOWN`` when it has more than one level, so a single-level
    run never visits ``IT_DOWN``, ``IT_COARSE`` or ``IT_UP``.
    """
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    visited = set()

    class Recorder(controller_nonMPI):
        def pfasst(self, local_MS_active):
            visited.update(S.status.stage for S in local_MS_active)
            return super().pfasst(local_MS_active)

    description = get_description('multi_level')
    controller = Recorder(num_procs=NUM_PROCS, controller_params={'logger_level': 30}, description=description)
    P = controller.MS[0].levels[0].prob
    controller.run(u0=P.u_exact(0.0), t0=0.0, Tend=NUM_PROCS * DT)

    assert {'IT_DOWN', 'IT_COARSE', 'IT_UP'} <= visited, f'missing stages, got {sorted(visited)}'


@pytest.mark.mpi4py
@pytest.mark.parametrize('case', ['baseline_single', 'baseline_multi'])
def test_pfasst_baseline(case, results):
    """Same description, same answer, whichever transport ran it."""
    serial, mpi = results[case]

    assert np.array_equal(
        serial['niter'], mpi['niter']
    ), f'iteration counts differ for {case}: {serial["niter"].tolist()} vs {mpi["niter"].tolist()}'
    assert np.allclose(
        serial['uend'], mpi['uend'], atol=ATOL, rtol=RTOL
    ), f'uend differs for {case} by {np.max(np.abs(serial["uend"] - mpi["uend"])):.3e} > {ATOL:.0e}'


@pytest.mark.mpi4py
def test_stats_emission_agrees(results):
    """A user's post-processing must not depend on which transport produced the stats."""
    serial, mpi = results['baseline_multi']
    serial_types, mpi_types = list(serial['stats_types']), [str(me) for me in mpi['stats_types']]

    assert serial_types == mpi_types, (
        f'stats types differ:\n  only serial: {sorted(set(serial_types) - set(mpi_types))}'
        f'\n  only MPI   : {sorted(set(mpi_types) - set(serial_types))}'
    )


@pytest.mark.mpi4py
def test_block_hook_fires_once_per_iteration(results):
    """
    ``post_iteration_processing_block`` must fire once per iteration, in both transports.

    This is the contract that lets a convergence controller hold block-global state at all. The
    per-step hooks fire once per *step*, so a controller is called L times per iteration virtually
    and once per iteration under MPI; anything recursive on ``self`` therefore advanced at a rate
    that depended only on how the run was decomposed. The block hook is the fix, and this is what
    makes it checkable.

    Note what is deliberately *not* asserted: that a recursion driven from the block hook reaches the
    same value in both transports. It does not, and should not. Under a pipelined method an early
    step converges and drops out, so ranks legitimately take part in different numbers of iterations
    -- ``niter`` here is [6, 7, 8, 9] across the block. That spread is the pipelining (C2), not
    drift. The invariant that survives it is the per-rank one asserted below, and it is exactly
    strong enough for the case that motivated the hook: in ParaDiag every step iterates together, so
    once-per-iteration-per-rank is once-per-iteration full stop, which is why ``AdaptiveAlpha`` gets
    identical answers either way (see ``test_adaptive_alpha.py`` and tutorial step_9 E, which pin
    that and are not duplicated here).
    """
    serial, mpi = results['probe']

    for name, r in (('virtual', serial), ('MPI', mpi)):
        block_calls, iters_seen = int(r['block_calls']), int(r['iters_seen'])
        assert block_calls > 1, f'the block hook never fired in the {name} controller'
        assert block_calls == iters_seen, (
            f'the block hook fired {block_calls} times over {iters_seen} iterations in the '
            f'{name} controller -- it must fire once per iteration, not once per step'
        )


@pytest.mark.mpi4py
def test_iteration_estimator_flag_treated_identically(results):
    """
    Both controllers must do the same thing with ``use_iteration_estimator``.

    They did not. It is declared in the *base* controller's parameters, so both accepted it, but only
    ``controller_MPI`` read it -- and that implementation was never switched on anywhere, had no
    tests, and deadlocked when used. ``controller_nonMPI`` ignored the flag in silence and ran to
    ``maxiter``. Same description, two different programs, no error either way.

    Both now refuse it and point at the ``CheckIterationEstimatorNonMPI`` convergence controller,
    which is the route that actually works. That is agreement, not yet parity: the convergence
    controller is still nonMPI-only, which is the next thing to fix. Refusing loudly is what makes
    the remaining gap visible instead of silent.
    """
    assert iteration_estimator_rejected(useMPI=False), 'controller_nonMPI accepted use_iteration_estimator'
    assert results['estimator_rejected_mpi'], 'controller_MPI accepted use_iteration_estimator'


@pytest.mark.mpi4py
@pytest.mark.parametrize('case', ['baseline_single', 'baseline_multi'])
def test_pipelining_is_preserved(case, results):
    """
    An early step has to converge and stop while later ones carry on. In both controllers.

    That spread *is* the pipelining. PFASST and MSSDC exist to let a step decide for itself, from its
    own values and its predecessor's status, so the running set shrinks and no rank waits on a
    neighbour that has already finished. A convergence criterion that stops every step at the same
    iteration has quietly turned the method into a lockstep one, and nothing else here would notice:
    the run still converges, the answer is still right, and both transports still agree with each
    other -- they would simply agree on the wrong thing.

    It is an easy property to lose. Anything that reduces over the whole block to decide convergence
    takes it away, and a global convergence criterion is often the more natural thing to write.

    Asserted for both transports, because losing it in one and not the other is the more likely
    accident and would show up here as a mismatch rather than as this.
    """
    serial, mpi = results[case]

    for name, r in (('virtual', serial), ('MPI', mpi)):
        niter = np.asarray(r['niter'])[:, 1]
        assert len(set(niter.tolist())) > 1, (
            f'every step of the {name} controller stopped at iteration {niter[0]:.0f}. '
            f'Convergence has become global and the pipeline is gone.'
        )


@pytest.mark.mpi4py
def test_global_convergence_stays_available(results):
    """
    ``all_to_done`` must keep doing what it says, in both controllers.

    Local convergence is the production default, but the global option is deliberately kept: it
    matches the theory more closely and is wanted for testing. So it has to stay reachable, and it
    has to mean the same thing either way -- which is also the control for the test above. If a run
    with ``all_to_done`` set did not stop every step together, that test would be passing for no
    reason.
    """
    serial, mpi = results['global_convergence']

    for name, r in (('virtual', serial), ('MPI', mpi)):
        niter = np.asarray(r['niter'])[:, 1]
        assert len(set(niter.tolist())) == 1, (
            f'the {name} controller ran `all_to_done` but its steps stopped at {niter.tolist()}, ' f'not together'
        )

    assert np.array_equal(serial['niter'], mpi['niter']), (
        f'`all_to_done` means different things in the two controllers: '
        f'{serial["niter"][:, 1].tolist()} vs {mpi["niter"][:, 1].tolist()}'
    )


@pytest.mark.mpi4py
def test_gauss_seidel_mssdc_agrees(results):
    """
    Gauss-Seidel MSSDC has to be the same program in both transports too.

    This is the only axis that runs ``mssdc_jac=False``, where a step takes its predecessor's
    *current* iterate rather than the previous one, so the steps converge strictly in turn and the
    running set drains one at a time. Nothing else here reaches that path: with Jacobi coupling the
    block stays whole until the end, so ``it_coarse`` is never the iteration.

    The second assertion is the point, and it is not an equality. When the running set reaches
    exactly one, the two controllers used to route that last step differently --
    ``controller_nonMPI`` asked ``len(local_MS_running) == 1``, which is knowledge no single rank
    has, while ``controller_MPI`` asked ``num_procs == 1`` -- so one went to ``IT_FINE`` and the
    other to ``IT_COARSE``. Both are now ``S.status.time_size == 1``, the block size either
    transport can read locally.

    Be clear about what that means for this axis: comparing the two transports would **not** have
    found that divergence, and does not defend against its return. By the time the set is down to
    one step, that step has no successor to send to and its predecessor is done, so ``IT_FINE`` and
    ``IT_COARSE`` both come down to a single ``update_nodes`` -- the two controllers ran different
    stages and got bit-identical answers. It is the same blind spot the module docstring describes
    for the pipelining, in a third form: not a property both sides lose at once, but a difference
    neither side expresses in its output.

    What the axis does buy is the path. ``it_coarse`` as the iteration, and a running set that
    drains one step at a time, are reached by nothing else in this file, so any *future* divergence
    there -- one that does change an answer -- now has a test looking at it. The last assertion
    keeps that true by pinning the configuration rather than the result: if the iteration counts
    stop ending on a strict increase, the block no longer drains to a single step and this has
    quietly stopped covering the case it was written for. Retune the problem; do not drop it.
    """
    serial, mpi = results['gauss_seidel']

    assert np.array_equal(
        serial['niter'], mpi['niter']
    ), f'iteration counts differ: {serial["niter"].tolist()} vs {mpi["niter"].tolist()}'
    assert np.allclose(
        serial['uend'], mpi['uend'], atol=ATOL, rtol=RTOL
    ), f'uend differs by {np.max(np.abs(serial["uend"] - mpi["uend"])):.3e} > {ATOL:.0e}'

    counts = serial['niter'][:, 1]
    assert counts[-1] > counts[-2], (
        'the block no longer drains to a single running step, so this axis has stopped covering '
        f'the routing it was written for -- iteration counts were {counts.tolist()}'
    )


if __name__ == '__main__':
    from mpi4py import MPI

    _out = sys.argv[1]
    _payload = {}
    for _name, _kwargs in CASES.items():
        _result = run(useMPI=True, **_kwargs)
        _payload.update({f'{_name}/{_k}': _v for _k, _v in _result.items()})
    _payload['estimator/rejected'] = iteration_estimator_rejected(useMPI=True)

    if MPI.COMM_WORLD.rank == 0:
        np.savez(_out, **_payload)
