"""
Driver run under ``mpirun`` to compare the node-parallel delta sweeper against the serial one.

Each rank owns one collocation node. Rank 0 compares the result against a serial reference computed
in the same process and exits non-zero on mismatch, so the spawning test only has to check the
return code.
"""

import sys

import numpy as np
from mpi4py import MPI

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.sweeper_classes.generic_implicit_MPI import generic_implicit_MPI
from pySDC.projects.DeltaSDC.sweepers import delta_implicit
from pySDC.projects.DeltaSDC.sweepers_MPI import delta_implicit_MPI

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


def run(sweeper_class, params):
    description = {
        'problem_class': heatNd_unforced,
        'problem_params': HEAT_PARAMS,
        'sweeper_class': sweeper_class,
        'sweeper_params': params,
        'level_params': {'restol': -1, 'dt': 1e-2},
        'step_params': {'maxiter': 6},
    }
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
        print('OK', flush=True)


if __name__ == '__main__':
    main()
