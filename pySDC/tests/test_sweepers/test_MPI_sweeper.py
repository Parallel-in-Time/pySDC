import pytest


def run(use_MPI, num_nodes, quad_type, residual_type, imex):
    """
    Run a single sweep for a problem and compute the solution at the end point with a sweeper as specified.

    Args:
        use_MPI (bool): Use the MPI version of the sweeper or not
        num_nodes (int): The number of nodes to use
        quad_type (str): Type of nodes
        residual_type (str): Type of residual computation
        imex (bool): Use IMEX sweeper or not

    Returns:
        pySDC.Level.level: The level containing relevant data
    """
    import numpy as np
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    if not imex:
        if use_MPI:
            from pySDC.implementations.sweeper_classes.generic_implicit_MPI import generic_implicit_MPI as sweeper_class
        else:
            from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class

        from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d as problem_class
    else:
        if use_MPI:
            from pySDC.implementations.sweeper_classes.imex_1st_order_MPI import imex_1st_order_MPI as sweeper_class
        else:
            from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order as sweeper_class

        from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced as problem_class

    dt = 1e-1
    sweeper_params = {'num_nodes': num_nodes, 'quad_type': quad_type, 'QI': 'IEpar', 'QE': 'PIC'}
    description = {}
    description['problem_class'] = problem_class
    description['sweeper_class'] = sweeper_class
    description['sweeper_params'] = sweeper_params
    description['level_params'] = {'dt': dt, 'residual_type': residual_type}
    description['step_params'] = {'maxiter': 1}

    controller = controller_nonMPI(1, {'logger_level': 30}, description)

    if imex:
        u0 = controller.MS[0].levels[0].prob.u_exact(0)
    else:
        u0 = np.ones_like(controller.MS[0].levels[0].prob.u_exact(0))
    controller.run(u0, 0, dt)
    controller.MS[0].levels[0].sweep.compute_end_point()
    return controller.MS[0].levels[0]


@pytest.mark.mpi4py
@pytest.mark.parametrize("num_nodes", [2])
@pytest.mark.parametrize("quad_type", ['GAUSS', 'RADAU-RIGHT'])
@pytest.mark.parametrize("residual_type", ['last_abs', 'full_rel'])
@pytest.mark.parametrize("imex", [True, False])
def test_sweeper(num_nodes, quad_type, residual_type, imex, launch=True):
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
    if launch:
        import os
        import subprocess

        # Set python path once
        my_env = os.environ.copy()
        my_env['PYTHONPATH'] = '../../..:.'
        my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

        cmd = f"mpirun -np {num_nodes} python {__file__} --test_sweeper {num_nodes} {quad_type} {residual_type} {imex}".split()

        p = subprocess.Popen(cmd, env=my_env, cwd=".")

        p.wait()
        assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (
            p.returncode,
            num_nodes,
        )
    else:
        import numpy as np

        imex = False if imex == 'False' else True
        MPI = run(use_MPI=True, num_nodes=int(num_nodes), quad_type=quad_type, residual_type=residual_type, imex=imex)
        nonMPI = run(
            use_MPI=False, num_nodes=int(num_nodes), quad_type=quad_type, residual_type=residual_type, imex=imex
        )

        assert np.allclose(MPI.uend, nonMPI.uend, atol=1e-14), 'Got different solutions at end point!'
        assert np.allclose(MPI.status.residual, nonMPI.status.residual, atol=1e-14), 'Got different residuals!'


if __name__ == '__main__':
    import sys

    if '--test_sweeper' in sys.argv:
        test_sweeper(sys.argv[-4], sys.argv[-3], sys.argv[-2], sys.argv[-1], launch=False)
