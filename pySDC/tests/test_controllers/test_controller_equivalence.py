"""
Characterisation test pinning `controller_nonMPI` and `controller_MPI` to the same answer.

The two controllers implement the same SDC/MLSDC/PFASST stage machine twice, once for all steps in
one process and once for one step per MPI rank. Nothing else in the test suite asserts that they
agree, so this file does: for the same `description`, `uend` and the per-step iteration counts must
match to `atol=1e-14, rtol=0`.

Note the tolerance is absolute only. `np.allclose` defaults to `rtol=1e-05`, which for values of
order one would swamp `atol=1e-14` completely, so `rtol=0` is passed explicitly everywhere.

The multi-level configuration is not optional: `controller_nonMPI.it_check` only sends a step to
`IT_DOWN` when `len(S.levels) > 1`, so a single-level run never visits `IT_DOWN`, `IT_COARSE` or
`IT_UP`. `test_multilevel_visits_all_stages` guards that the two-level configuration really does.

The file doubles as the MPI entry point: the pytest process shells out to
`mpirun -np N python <this file> ...`, the subprocess writes its result to an npz, and the serial
run happens in-process for comparison.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

ATOL = 1e-14
RTOL = 0.0
DT = 0.1

CONFIGS = ['single_level', 'multi_level']


def get_description(config):
    """Single-level IMEX SDC, or the same problem as two-level MLSDC/PFASST."""
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced
    from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
    from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh

    description = {
        'problem_class': heatNd_forced,
        'problem_params': {'nu': 0.1, 'freq': (2,), 'nvars': (31,), 'bc': 'dirichlet-zero'},
        'sweeper_class': imex_1st_order,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'IE'},
        'level_params': {'restol': 1e-11, 'dt': DT},
        'step_params': {'maxiter': 20},
    }

    if config == 'multi_level':
        description['problem_params']['nvars'] = [(63,), (31,)]
        description['sweeper_params']['num_nodes'] = [3, 2]
        description['space_transfer_class'] = mesh_to_mesh
        description['space_transfer_params'] = {'rorder': 2, 'iorder': 6}

    return description


def run(config, num_procs, useMPI, controller_class=None):
    """
    Run one configuration through either controller and return what we compare.

    Returns:
        numpy.ndarray: solution at the end of the block
        numpy.ndarray: (time, niter) pairs, gathered across ranks under MPI
    """
    from pySDC.helpers.stats_helper import get_sorted

    description = get_description(config)
    controller_params = {'logger_level': 30}

    if useMPI:
        from mpi4py import MPI
        from pySDC.implementations.controller_classes.controller_MPI import controller_MPI

        comm = MPI.COMM_WORLD
        controller = (controller_class or controller_MPI)(
            comm=comm, controller_params=controller_params, description=description
        )
        P = controller.S.levels[0].prob
    else:
        from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

        comm = None
        controller = (controller_class or controller_nonMPI)(
            num_procs=num_procs, controller_params=controller_params, description=description
        )
        P = controller.MS[0].levels[0].prob

    t0 = 0.0
    uend, stats = controller.run(u0=P.u_exact(t0), t0=t0, Tend=num_procs * DT)

    # stats are rank-local under MPI, so gather them before comparing
    niter = get_sorted(stats, type='niter', sortby='time', comm=comm)

    return np.asarray(uend), np.array(niter, dtype=float)


@pytest.mark.base
@pytest.mark.parametrize('num_procs', [1, 4])
def test_multilevel_visits_all_stages(num_procs):
    """Without this, a green equivalence test could still be covering only 4 of the 7 stages."""
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    visited = set()

    class Recorder(controller_nonMPI):
        def pfasst(self, local_MS_active):
            visited.update(S.status.stage for S in local_MS_active)
            return super().pfasst(local_MS_active)

    run('multi_level', num_procs, useMPI=False, controller_class=Recorder)

    assert {'IT_DOWN', 'IT_COARSE', 'IT_UP'} <= visited, f'missing stages, got {sorted(visited)}'


@pytest.mark.mpi4py
@pytest.mark.parametrize('config', CONFIGS)
@pytest.mark.parametrize('num_procs', [1, 4])
def test_controller_equivalence(config, num_procs, tmp_path):
    """The point of the file: same description, same answer, whichever controller ran it."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    result_file = str(tmp_path / 'mpi_result.npz')

    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = root
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

    cmd = f'mpirun -np {num_procs} {sys.executable} {__file__} {config} {num_procs} {result_file}'.split()
    p = subprocess.Popen(cmd, env=my_env, cwd=root)
    p.wait()
    assert p.returncode == 0, f'ERROR: mpirun returned {p.returncode} with {num_procs} processes'

    mpi_result = np.load(result_file)
    uend, niter = run(config, num_procs, useMPI=False)

    assert np.array_equal(niter, mpi_result['niter']), (
        f'iteration counts differ for {config} on {num_procs} processes: '
        f'serial {niter.tolist()} vs. MPI {mpi_result["niter"].tolist()}'
    )
    assert np.allclose(uend, mpi_result['uend'], atol=ATOL, rtol=RTOL), (
        f'uend differs for {config} on {num_procs} processes by '
        f'{np.max(np.abs(uend - mpi_result["uend"])):.3e} > {ATOL:.0e}'
    )


if __name__ == '__main__':
    from mpi4py import MPI

    _config, _num_procs, _result_file = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    _uend, _niter = run(_config, _num_procs, useMPI=True)

    if MPI.COMM_WORLD.rank == 0:
        np.savez(_result_file, uend=_uend, niter=_niter)
