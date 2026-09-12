r"""
Tests for :mod:`pySDC.implementations.transfer_classes.BaseTransferDeltaMPI`.

One collocation node per rank, so the node-parallel hierarchy has to reproduce the serial one. Both
identities it rests on become reductions here -- the restricted fine residual and the accumulated
coarse correction each couple all nodes -- which is the part that can be got wrong without changing
any answer at backend precision on a single rank.

Follows the launch pattern of ``test_MPI_sweeper.py``: pytest re-executes this module under
``mpirun``, and the ``__main__`` block below runs the comparison inside it.
"""

import pytest

SWEEPER_PARAMS = {
    'quad_type': 'RADAU-RIGHT',
    'node_type': 'LEGENDRE',
    'QI': 'MIN-SR-S',  # the node-parallel layout uses only the diagonal of QI, so it has to be one
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

NVARS = [63, 31]


def run(use_MPI, num_nodes, stock_hierarchy):
    """Run a two-level V-cycle and return the end value."""
    from pySDC.core.base_transfer import BaseTransfer
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh

    sweeper_params = dict(SWEEPER_PARAMS, num_nodes=num_nodes)

    if use_MPI:
        from mpi4py import MPI
        from pySDC.implementations.sweeper_classes.delta_form_MPI import delta_implicit_MPI as sweeper_class
        from pySDC.implementations.transfer_classes.BaseTransferMPI import base_transfer_MPI
        from pySDC.implementations.transfer_classes.BaseTransferDeltaMPI import delta_transfer_MPI

        sweeper_params['comm'] = MPI.COMM_WORLD
        transfer_class = base_transfer_MPI if stock_hierarchy else delta_transfer_MPI
    else:
        from pySDC.implementations.sweeper_classes.delta_form import delta_implicit as sweeper_class
        from pySDC.implementations.transfer_classes.BaseTransferDelta import delta_transfer

        transfer_class = BaseTransfer if stock_hierarchy else delta_transfer

    description = {
        'problem_class': heatNd_unforced,
        'problem_params': dict(HEAT_PARAMS, nvars=NVARS),
        'sweeper_class': sweeper_class,
        'sweeper_params': sweeper_params,
        'level_params': {'restol': -1, 'dt': 1e-2},
        'step_params': {'maxiter': 6},
        'space_transfer_class': mesh_to_mesh,
        'space_transfer_params': {'iorder': 4, 'rorder': 2},
        'base_transfer_class': transfer_class,
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    prob = controller.MS[0].levels[0].prob
    uend, _ = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=2e-2)
    return uend


def individual_test(num_nodes, launch=False):
    """Compare the node-parallel delta hierarchy against the serial one, or launch mpirun to do so."""
    if launch:
        import os
        import subprocess

        my_env = os.environ.copy()
        my_env['PYTHONPATH'] = '../../..:.'
        my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

        cmd = f'mpirun -np {num_nodes} python {__file__} --num_nodes={num_nodes}'
        p = subprocess.Popen(cmd.split(), env=my_env, cwd='.')
        p.wait()
        assert p.returncode == 0, f'got return code {p.returncode} with {num_nodes} processes'
        return

    parallel = run(True, num_nodes, stock_hierarchy=False)
    serial = run(False, num_nodes, stock_hierarchy=False)
    stock = run(True, num_nodes, stock_hierarchy=True)

    assert (
        abs(parallel - serial) < 1e-13
    ), f'node-parallel and serial delta hierarchy disagree by {abs(parallel - serial):.3e}'
    # and against the stock node-parallel hierarchy, which is what says the two identities hold
    # in their reduced form rather than merely agreeing with each other
    assert abs(parallel - stock) < 1e-13, f'the delta hierarchy moved the answer by {abs(parallel - stock):.3e}'


@pytest.mark.mpi4py
@pytest.mark.parametrize('num_nodes', [2, 3])
def test_matches_the_serial_and_the_stock_hierarchy(num_nodes):
    """
    The reduced identities have to give what the serial loops give, and what stock MLSDC gives.

    Comparing only against the serial delta hierarchy would pass if both were wrong the same way,
    which is why the stock node-parallel hierarchy is the third leg.
    """
    individual_test(num_nodes, launch=True)


if __name__ == '__main__':
    import sys

    kwargs = dict(arg.removeprefix('--').split('=') for arg in sys.argv[1:])
    individual_test(num_nodes=int(kwargs['num_nodes']))
