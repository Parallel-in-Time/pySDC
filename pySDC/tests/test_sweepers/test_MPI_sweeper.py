import pytest


def run(use_MPI, num_nodes, quad_type, residual_type, imex, init_guess, useNCCL, ML):
    """
    Run a single sweep for a problem and compute the solution at the end point with a sweeper as specified.

    Args:
        use_MPI (bool): Use the MPI version of the sweeper or not
        num_nodes (int): The number of nodes to use
        quad_type (str): Type of nodes
        residual_type (str): Type of residual computation
        imex (bool): Use IMEX sweeper or not
        init_guess (str): which initial guess should be used
        useNCCL (bool): ...
        ML (int): Number of levels in space

    Returns:
        pySDC.Level.level: The level containing relevant data
    """
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    if not imex:
        if use_MPI:
            from pySDC.implementations.sweeper_classes.generic_implicit_MPI import generic_implicit_MPI as sweeper_class
        else:
            from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class

        if ML:
            from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced as problem_class
        else:
            from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d as problem_class
    else:
        if use_MPI:
            from pySDC.implementations.sweeper_classes.imex_1st_order_MPI import imex_1st_order_MPI as sweeper_class
        else:
            from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order as sweeper_class

        from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced as problem_class

    dt = 1e-1
    description = {}
    sweeper_params = {
        'num_nodes': num_nodes,
        'quad_type': quad_type,
        'QI': 'IEpar',
        'QE': 'PIC',
        "initial_guess": init_guess,
    }
    problem_params = {}

    if useNCCL:
        from pySDC.helpers.NCCL_communicator import NCCLComm
        from mpi4py import MPI

        sweeper_params['comm'] = NCCLComm(MPI.COMM_WORLD)
        problem_params['useGPU'] = True

    if ML > 1:
        from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh

        description['space_transfer_class'] = mesh_to_mesh

        problem_params['nvars'] = [2 ** (ML - i) for i in range(ML)]
        if use_MPI:
            from pySDC.implementations.transfer_classes.BaseTransferMPI import base_transfer_MPI

            description['base_transfer_class'] = base_transfer_MPI

    description['problem_class'] = problem_class
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_class
    description['sweeper_params'] = sweeper_params
    description['level_params'] = {'dt': dt, 'residual_type': residual_type}
    description['step_params'] = {'maxiter': 1}

    controller = controller_nonMPI(1, {'logger_level': 30}, description)

    if imex:
        u0 = controller.MS[0].levels[0].prob.u_exact(0)
    else:
        u0 = controller.MS[0].levels[0].prob.u_exact(0) + 1.0
    controller.run(u0, 0, dt)
    controller.MS[0].levels[0].sweep.compute_end_point()
    return controller.MS[0].levels[0]


def individual_test(launch=False, **kwargs):
    """
    Make a test if the result matches between the MPI and non-MPI versions of a sweeper.
    Tests solution at the right end point and the residual.

    Args:
        launch (bool): If yes, it will launch `mpirun` with the required number of processes
    """
    num_nodes = kwargs['num_nodes']
    useNCCL = kwargs['useNCCL']

    if launch:
        import os
        import subprocess

        # Set python path once
        my_env = os.environ.copy()
        my_env['PYTHONPATH'] = '../../..:.'
        my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

        cmd = f"mpirun -np {num_nodes} python {__file__}"

        for key, value in kwargs.items():
            cmd += f' --{key}={value}'
        p = subprocess.Popen(cmd.split(), env=my_env, cwd=".")

        p.wait()
        assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (
            p.returncode,
            num_nodes,
        )
    else:
        if useNCCL:
            import cupy as xp
        else:
            import numpy as xp

        MPI = run(
            **kwargs,
            use_MPI=True,
        )
        nonMPI = run(
            **kwargs,
            use_MPI=False,
        )

        assert xp.allclose(
            MPI.uend, nonMPI.uend, atol=1e-14
        ), f'Got different solutions at end point! {MPI.uend=} {nonMPI.uend=}'
        assert xp.allclose(MPI.status.residual, nonMPI.status.residual, atol=1e-14), 'Got different residuals!'


@pytest.mark.mpi4py
@pytest.mark.parametrize("num_nodes", [2])
@pytest.mark.parametrize("quad_type", ['GAUSS', 'RADAU-RIGHT'])
@pytest.mark.parametrize("residual_type", ['last_abs', 'full_rel'])
@pytest.mark.parametrize("imex", [True, False])
@pytest.mark.parametrize("init_guess", ['spread', 'copy', 'zero'])
@pytest.mark.parametrize("ML", [1, 2, 3])
def test_sweeper(num_nodes, quad_type, residual_type, imex, init_guess, ML, launch=True):
    """
    Make a test if the result matches between the MPI and non-MPI versions of a sweeper.
    Tests solution at the right end point and the residual.

    Args:
        num_nodes (int): The number of nodes to use
        quad_type (str): Type of nodes
        residual_type (str): Type of residual computation
        imex (bool): Use IMEX sweeper or not
        launch (bool): If yes, it will launch `mpirun` with the required number of processes
    """
    individual_test(
        num_nodes=num_nodes,
        quad_type=quad_type,
        residual_type=residual_type,
        imex=imex,
        init_guess=init_guess,
        useNCCL=False,
        ML=ML,
        launch=launch,
    )


@pytest.mark.cupy
@pytest.mark.skip(reason="We haven\'t figured out how to run tests on the cluster with multiple processes yet.")
@pytest.mark.parametrize("num_nodes", [2])
@pytest.mark.parametrize("quad_type", ['GAUSS', 'RADAU-RIGHT'])
@pytest.mark.parametrize("residual_type", ['last_abs', 'full_rel'])
@pytest.mark.parametrize("imex", [False])
@pytest.mark.parametrize("init_guess", ['spread', 'copy', 'zero'])
def test_sweeper_NCCL(num_nodes, quad_type, residual_type, imex, init_guess, launch=True):
    """
    Make a test if the result matches between the MPI and non-MPI versions of a sweeper.
    Tests solution at the right end point and the residual.

    Args:
        num_nodes (int): The number of nodes to use
        quad_type (str): Type of nodes
        residual_type (str): Type of residual computation
        imex (bool): Use IMEX sweeper or not
        launch (bool): If yes, it will launch `mpirun` with the required number of processes
    """
    individual_test(
        num_nodes=num_nodes,
        quad_type=quad_type,
        residual_type=residual_type,
        imex=imex,
        init_guess=init_guess,
        useNCCL=True,
        ML=1,
        launch=launch,
    )


if __name__ == '__main__':
    str_to_bool = lambda me: False if me == 'False' else True
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ML', type=int, help='Number of levels in space')
    parser.add_argument('--num_nodes', type=int, help='Number of collocation nodes')
    parser.add_argument('--quad_type', type=str, help='Quadrature rule', choices=['GAUSS', 'RADAU-RIGHT', 'RADAU-LEFT'])
    parser.add_argument(
        '--residual_type',
        type=str,
        help='Way of computing the residual',
        choices=['full_rel', 'last_abs', 'full_abs', 'last_rel'],
    )
    parser.add_argument('--imex', type=str_to_bool, help='Toggle for IMEX', choices=[True, False])
    parser.add_argument('--useNCCL', type=str_to_bool, help='Toggle for NCCL communicator', choices=[True, False])
    parser.add_argument('--init_guess', type=str, help='Initial guess', choices=['spread', 'copy', 'zero'])
    args = parser.parse_args()

    individual_test(**vars(args))
