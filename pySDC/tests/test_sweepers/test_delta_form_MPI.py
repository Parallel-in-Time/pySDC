r"""
Tests for :mod:`pySDC.implementations.sweeper_classes.delta_form_MPI`.

One collocation node per rank, so the node-parallel delta form has to reproduce the serial one
exactly. The MPI-specific piece is the scale ``correction_precision`` quantises against: it is a
property of the whole sweep, but each rank sees only its own node's residual, so it has to be
reduced. Left un-reduced, every rank quantises against a different divisor and the run silently
stops matching the serial one -- which is why the reduced-precision case is parametrised here.

Follows the launch pattern of ``test_MPI_sweeper.py``: pytest re-executes this module under
``mpirun``, and the ``__main__`` block below runs the comparison inside it.
"""

import pytest

SWEEPER_PARAMS = {
    'quad_type': 'RADAU-RIGHT',
    'node_type': 'LEGENDRE',
    'QI': 'IEpar',  # the node-parallel sweeper uses only the diagonal of QI, so it has to be diagonal
    'initial_guess': 'spread',
    'linear_implicit': True,
}

HEAT_PARAMS = {
    'nvars': 63,
    'nu': 1.0,
    'freq': 2,
    'bc': 'dirichlet-zero',
    'order': 2,
    'solver_type': 'direct',
}


def run(use_MPI, num_nodes, correction_precision):
    """Run four sweeps and return the finest level."""
    import numpy as np
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced

    sweeper_params = dict(SWEEPER_PARAMS, num_nodes=num_nodes)
    if correction_precision != 'None':
        sweeper_params['correction_precision'] = np.dtype(correction_precision)

    if use_MPI:
        from mpi4py import MPI
        from pySDC.implementations.sweeper_classes.delta_form_MPI import delta_implicit_MPI as sweeper_class

        sweeper_params['comm'] = MPI.COMM_WORLD
    else:
        from pySDC.implementations.sweeper_classes.delta_form import delta_implicit as sweeper_class

    description = {
        'problem_class': heatNd_unforced,
        'problem_params': HEAT_PARAMS,
        'sweeper_class': sweeper_class,
        'sweeper_params': sweeper_params,
        'level_params': {'restol': -1, 'dt': 1e-2},
        'step_params': {'maxiter': 4},
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    prob = controller.MS[0].levels[0].prob
    controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=2e-2)
    return controller.MS[0].levels[0]


def individual_test(num_nodes, correction_precision, launch=False):
    """Compare the node-parallel delta form against the serial one, or launch mpirun to do so."""
    if launch:
        import os
        import subprocess

        my_env = os.environ.copy()
        my_env['PYTHONPATH'] = '../../..:.'
        my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

        cmd = f'mpirun -np {num_nodes} python {__file__}'
        cmd += f' --num_nodes={num_nodes} --correction_precision={correction_precision}'
        p = subprocess.Popen(cmd.split(), env=my_env, cwd='.')
        p.wait()
        assert p.returncode == 0, f'got return code {p.returncode} with {num_nodes} processes'
        return

    parallel = run(True, num_nodes, correction_precision)
    serial = run(False, num_nodes, correction_precision)

    assert abs(parallel.uend - serial.uend) < 1e-14, (
        f'node-parallel and serial delta form disagree at the end point by ' f'{abs(parallel.uend - serial.uend):.3e}'
    )
    assert abs(parallel.status.residual - serial.status.residual) < 1e-14, 'the residuals disagree'


@pytest.mark.mpi4py
@pytest.mark.parametrize('num_nodes', [2, 3])
@pytest.mark.parametrize('correction_precision', ['None', 'float32'])
def test_matches_the_serial_delta_form(num_nodes, correction_precision):
    """
    The node-parallel sweep is the same sweep, so it must give the same answer.

    ``float32`` corrections are the case that needs the reduced scale: each rank holds one node and
    so sees only part of the residual the divisor is taken from.
    """
    individual_test(num_nodes, correction_precision, launch=True)


if __name__ == '__main__':
    import sys

    kwargs = {}
    for arg in sys.argv[1:]:
        key, value = arg.split('=')
        kwargs[key.removeprefix('--')] = value
    individual_test(num_nodes=int(kwargs['num_nodes']), correction_precision=kwargs['correction_precision'])
