"""
Node-parallel delta-form sweeper, run under ``mpirun`` with one rank per collocation node.

Runnable entry point in the style of ``pySDC/tutorial/step_7/C_pySDC_with_PETSc.py``: the
mpi4py-marked test spawns this with ``mpirun -np 3`` and only checks the return code.

Rank 0 compares against a serial reference computed in the same process and aborts on mismatch.

Usage::

    mpirun -np 3 python pySDC/projects/DeltaSDC/run_mpi.py [--fp32]
"""

import sys

import numpy as np
from mpi4py import MPI

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.sweeper_classes.generic_implicit_MPI import generic_implicit_MPI
from pySDC.implementations.transfer_classes.BaseTransferMPI import base_transfer_MPI
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.projects.DeltaSDC.mlsdc import delta_implicit_rounded, delta_transfer
from pySDC.projects.DeltaSDC.problems import heat_delta
from pySDC.implementations.sweeper_classes.delta_form import delta_implicit
from pySDC.implementations.sweeper_classes.delta_form_MPI import delta_implicit_MPI
from pySDC.projects.DeltaSDC.sweepers_MPI import (
    delta_implicit_MPI_cascade,
    delta_implicit_MPI_rounded,
    delta_transfer_MPI,
)

COARSE_NVARS = 31

HEAT_PARAMS = {
    'nvars': 63,
    'nu': 1.0,
    'freq': 2,
    'bc': 'dirichlet-zero',
    'order': 2,
    'solver_type': 'direct',
}


def sweeper_params(comm=None, **extra):
    params = {
        'quad_type': 'RADAU-RIGHT',
        'node_type': 'LEGENDRE',
        'num_nodes': 3,
        'QI': 'MIN-SR-S',  # diagonal, required for node parallelism
        'initial_guess': 'spread',
    }
    if comm is not None:
        params['comm'] = comm
    params.update(extra)
    return params


def run(sweeper_class, params, problem_class=heatNd_unforced, multilevel=False, base_transfer_class=None):
    description = {
        'problem_class': problem_class,
        'problem_params': HEAT_PARAMS,
        'sweeper_class': sweeper_class,
        'sweeper_params': params,
        'level_params': {'restol': -1, 'dt': 1e-2},
        'step_params': {'maxiter': 6},
    }
    if multilevel:
        description['problem_params'] = dict(HEAT_PARAMS, nvars=[HEAT_PARAMS['nvars'], COARSE_NVARS])
        description['space_transfer_class'] = mesh_to_mesh
        description['space_transfer_params'] = {'iorder': 4, 'rorder': 2}
        description['base_transfer_class'] = base_transfer_class
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 40}, description=description)
    prob = controller.MS[0].levels[0].prob
    uend, _ = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=2e-2)
    return uend


def main():
    comm = MPI.COMM_WORLD
    if comm.size != 3:
        raise RuntimeError(f'this driver needs one rank per collocation node, got {comm.size}')

    precision = np.dtype('float32') if '--fp32' in sys.argv else None
    extra = {'correction_precision': precision} if precision is not None else {}

    # Comparing two runs at the SAME precision must be exact to round-off. Comparing against a
    # full-precision reference must allow the reduced-precision perturbation of the corrections.
    same_precision_tol = 1e-11
    cross_precision_tol = 1e-11 if precision is None else 1e-9

    parallel_delta = run(delta_implicit_MPI, sweeper_params(comm=comm, **extra))

    if comm.rank == 0:
        serial_delta = run(delta_implicit, sweeper_params(**extra))
        serial_stock = run(generic_implicit, sweeper_params())

        errors = {
            'delta_MPI vs delta serial': (abs(parallel_delta - serial_delta), same_precision_tol),
            'delta_MPI vs generic_implicit': (abs(parallel_delta - serial_stock), cross_precision_tol),
        }
        for label, (value, tol) in errors.items():
            print(f'{label}: {value:.3e} (tol {tol:.0e})', flush=True)
        if any(value > tol for value, tol in errors.values()):
            print('MISMATCH', flush=True)
            comm.Abort(1)

    # cross-check that the stock MPI sweeper agrees too, i.e. the comparison itself is sound
    parallel_stock = run(generic_implicit_MPI, sweeper_params(comm=comm))
    if comm.rank == 0:
        diff = abs(parallel_delta - parallel_stock)
        print(f'delta_MPI vs generic_implicit_MPI: {diff:.3e} (tol {cross_precision_tol:.0e})', flush=True)
        if diff > cross_precision_tol:
            print('MISMATCH', flush=True)
            comm.Abort(1)

    check_multilevel(comm, extra)

    if comm.rank == 0:
        print('OK', flush=True)


def check_multilevel(comm, extra):
    """
    The same comparison one level down: node-parallel MLSDC, in the delta form and stock.

    ``base_transfer_MPI`` and :class:`delta_transfer_MPI` both require one rank per collocation
    node on *every* level, so the hierarchy coarsens in space only, which is the recipe that pays
    anyway.

    Parameters
    ----------
    comm : mpi4py.MPI.Intracomm
        The node communicator, one rank per collocation node.
    extra : dict
        Extra sweeper parameters, carrying ``correction_precision`` when ``--fp32`` was passed.

    Returns
    -------
    None
    """
    # linear_implicit reaches the correction equation through the stock solve_system, which this
    # problem's homogeneous Dirichlet operator allows. A reduced-precision level *needs* a genuine
    # correction route: the substitution fallback is exact but reads the level's O(1) state, which
    # is merely no benefit at backend precision and a wrong answer below it (1.5e-04 here).
    delta = {'linear_implicit': True}
    parallel = run(
        delta_implicit_MPI_rounded,
        sweeper_params(comm=comm, **delta, **extra),
        problem_class=heat_delta,
        multilevel=True,
        base_transfer_class=delta_transfer_MPI,
    )
    # the coarse level, all of it, at half precision -- which needs the delta hierarchy to be safe
    parallel_fp16 = run(
        delta_implicit_MPI_rounded,
        sweeper_params(comm=comm, level_precision=[None, np.float16], **delta, **extra),
        problem_class=heat_delta,
        multilevel=True,
        base_transfer_class=delta_transfer_MPI,
    )
    parallel_stock = run(
        generic_implicit_MPI, sweeper_params(comm=comm), multilevel=True, base_transfer_class=base_transfer_MPI
    )
    # storage precision raised as the iteration converges, with the indicator reduced across the
    # node communicator so every rank steps on the same sweep
    parallel_cascade = run(
        delta_implicit_MPI_cascade,
        sweeper_params(comm=comm, state_cascade=('float16', 'float32', None), **delta, **extra),
        problem_class=heat_delta,
        multilevel=True,
        base_transfer_class=delta_transfer_MPI,
    )

    if comm.rank != 0:
        return

    serial = run(
        delta_implicit_rounded,
        sweeper_params(**delta, **extra),
        problem_class=heat_delta,
        multilevel=True,
        base_transfer_class=delta_transfer,
    )
    errors = {
        'deltaMLSDC_MPI vs deltaMLSDC serial': (abs(parallel - serial), 1e-11),
        'deltaMLSDC_MPI vs stock MLSDC_MPI': (abs(parallel - parallel_stock), 1e-11),
        'deltaMLSDC_MPI fp16 coarse vs fp64': (abs(parallel_fp16 - parallel), 1e-11),
        'deltaMLSDC_MPI cascade vs fp64': (abs(parallel_cascade - parallel), 1e-9),
    }
    for label, (value, tol) in errors.items():
        print(f'{label}: {value:.3e} (tol {tol:.0e})', flush=True)
    if any(value > tol for value, tol in errors.values()):
        print('MISMATCH', flush=True)
        comm.Abort(1)


if __name__ == '__main__':
    main()
