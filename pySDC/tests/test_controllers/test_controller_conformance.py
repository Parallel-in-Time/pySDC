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

Axes marked ``xfail`` are known asymmetries with a documented cause. They are strict, so fixing one
without removing its marker fails the suite: the marker is a to-do list, not a suppression.

A note on the iteration estimator, since that asymmetry is not the obvious shape.
``controller_MPI`` implements it inline, ``controller_nonMPI`` not at all, and the supported
route is a third thing -- the ``CheckIterationEstimatorNonMPI`` convergence controller, which
tutorial step_8 C exercises. Only that third one is tested: nothing in the repository ever sets
``use_iteration_estimator`` to ``True``, and switching it on under MPI deadlocks. So this file
covers the virtual half of that axis only and never launches the MPI implementation.
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


def get_description(config, errtol=None):
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
    if errtol is not None:
        description['step_params']['errtol'] = errtol
    return description


def run(useMPI, config='single_level', probe=False, errtol=None, num_procs=NUM_PROCS):
    """
    Run one configuration through one of the two transports and return everything the axes compare.

    Returns:
        dict: uend, the (time, niter) pairs, the sorted stats types, and the probe's final state
    """
    from pySDC.helpers.stats_helper import get_sorted

    description = get_description(config, errtol=errtol)
    controller_params = {'logger_level': 30}
    if errtol is not None:
        controller_params['use_iteration_estimator'] = True
        controller_params['all_to_done'] = False
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

    uend, stats = controller.run(u0=P.u_exact(0.0), t0=0.0, Tend=num_procs * DT)

    probes = [me for me in controller.convergence_controllers if type(me) == RecursionProbe]
    return {
        'uend': np.asarray(uend),
        'niter': np.array(get_sorted(stats, type='niter', sortby='time', comm=comm), dtype=float),
        'stats_types': sorted({me.type for me in stats.keys()}),
        'block_calls': probes[0].block_calls if probes else -1,
        'iters_seen': len({it for _, it in probes[0].calls}) if probes else -1,
    }


CASES = {
    'baseline_single': {'config': 'single_level'},
    'baseline_multi': {'config': 'multi_level'},
    'probe': {'config': 'single_level', 'probe': True},
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


@pytest.mark.base
@pytest.mark.xfail(strict=True, reason='use_iteration_estimator is implemented in controller_MPI only')
def test_iteration_estimator_is_honoured():
    """
    ``use_iteration_estimator`` must do something, whichever controller is asked.

    It is declared in the base controller's parameters, so *both* controllers accept it, but only
    ``controller_MPI`` reads it. ``controller_nonMPI`` silently ignores the request and runs to
    ``maxiter``, with no error and no warning -- the same description, two different programs.

    This asks only whether the flag changes the virtual controller's behaviour, and deliberately
    does not run the MPI implementation: switching that one on deadlocks (see the module
    docstring), and a hanging test is worse than a missing one.
    """
    without = run(useMPI=False)['niter']
    with_estimator = run(useMPI=False, errtol=1e-7)['niter']

    assert not np.array_equal(without, with_estimator), (
        'use_iteration_estimator=True changed nothing in controller_nonMPI: '
        f'{without[:, 1].tolist()} iterations either way'
    )


if __name__ == '__main__':
    from mpi4py import MPI

    _out = sys.argv[1]
    _payload = {}
    for _name, _kwargs in CASES.items():
        _result = run(useMPI=True, **_kwargs)
        _payload.update({f'{_name}/{_k}': _v for _k, _v in _result.items()})

    if MPI.COMM_WORLD.rank == 0:
        np.savez(_out, **_payload)
