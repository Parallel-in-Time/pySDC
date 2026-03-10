import pytest


def run_simulation(sweeper_class, nsteps, nsweeps, nnodes, sweeper_comm=None):
    from mpi4py import MPI

    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.hooks.log_work import LogWork

    from pySDC.projects.RayleighBenard.tests.test_RBC_3D_analysis import get_args
    from pySDC.projects.RayleighBenard.RBC3D_configs import get_config

    args = get_args('.')
    if sweeper_comm is not None:
        args['procs'] = [1, sweeper_comm.size, 1]
    config = get_config(args)

    description = config.get_description(res=8)
    description['level_params']['nsweeps'] = nsweeps
    description['sweeper_params']['num_nodes'] = nnodes
    description['sweeper_class'] = sweeper_class

    controller_params = config.get_controller_params()
    controller_params['hook_class'] = [LogWork]

    controller = controller_nonMPI(1, controller_params, description)

    u0 = controller.MS[0].levels[0].prob.u_exact(0)
    dt = description['level_params']['dt']

    return controller.run(u0, 0, nsteps * dt)


@pytest.mark.parametrize('nsweeps', [1, 2, 3])
@pytest.mark.parametrize('nnodes', [1, 2, 3])
def test_serial_optimized_sweeper(nsweeps, nnodes):
    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
    from pySDC.projects.RayleighBenard.sweepers import imex_1st_order_diagonal_serial

    args = {
        'nsteps': 4,
        'nsweeps': nsweeps,
        'nnodes': nnodes,
    }

    u_opt, stats_opt = run_simulation(imex_1st_order_diagonal_serial, **args)
    u_normal, stats_normal = run_simulation(imex_1st_order, **args)

    assert np.allclose(u_normal, u_opt), 'Got different result with optimized sweeper'

    expect_work = (nsweeps - 1) * nnodes + 1
    got_rhs = [me[1] for me in get_sorted(stats_opt, type='work_rhs')]
    assert np.allclose(expect_work, got_rhs), f'Expected {expect_work} right hand side evaluations, but did {got_rhs}'

    got_solves = [me[1] for me in get_sorted(stats_opt, type='work_cached_direct')]
    assert np.allclose(expect_work, got_solves), f'Expected {expect_work} solves, but did {got_solves}'


@pytest.mark.parametrize('sweeper_name', ['ARK3', 'IMEXEulerStifflyAccurate'])
def test_RK_sweeper(sweeper_name):
    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
    from pySDC.implementations.sweeper_classes.Runge_Kutta import ARK3, IMEXEulerStifflyAccurate
    from pySDC.projects.RayleighBenard.sweepers import imex_1st_order_diagonal_serial

    sweepers = {'ARK3': ARK3, 'IMEXEulerStifflyAccurate': IMEXEulerStifflyAccurate}
    expected_work = {'ARK3': 4, 'IMEXEulerStifflyAccurate': 1}

    RK_sweeper_class = sweepers[sweeper_name]

    u, stats = run_simulation(RK_sweeper_class, 4, 1, 1)

    got_rhs = [me[1] for me in get_sorted(stats, type='work_rhs')]
    assert np.allclose(
        expected_work[sweeper_name], got_rhs
    ), f'Expected {expected_work[sweeper_name]} right hand side evaluations, but did {got_rhs}'

    got_solves = [me[1] for me in get_sorted(stats, type='work_cached_direct')]
    assert np.allclose(
        expected_work[sweeper_name], got_solves
    ), f'Expected {expected_work[sweeper_name]} solves, but did {got_solves}'


@pytest.mark.parametrize('nsweeps', [1, 2, 3])
@pytest.mark.mpi(ranks=[1, 2, 3])
def test_parallel_sweeper(mpi_ranks, nsweeps):
    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.sweeper_classes.imex_1st_order_MPI import imex_1st_order_MPI
    from pySDC.projects.RayleighBenard.sweepers import imex_1st_order_MPI_fixed_k
    from mpi4py import MPI

    args = {
        'nsteps': 4,
        'nsweeps': nsweeps,
        'nnodes': MPI.COMM_WORLD.size,
        'sweeper_comm': MPI.COMM_WORLD,
    }

    u_opt, stats_opt = run_simulation(imex_1st_order_MPI_fixed_k, **args)
    u_normal, stats_normal = run_simulation(imex_1st_order_MPI, **args)

    assert np.allclose(u_normal, u_opt), 'Got different result with optimized sweeper'

    expect_work = nsweeps
    got_rhs = [me[1] for me in get_sorted(stats_opt, type='work_rhs')]
    assert np.allclose(expect_work, got_rhs), f'Expected {expect_work} right hand side evaluations, but did {got_rhs}'

    got_solves = [me[1] for me in get_sorted(stats_opt, type='work_cached_direct')]
    assert np.allclose(expect_work, got_solves), f'Expected {expect_work} solves, but did {got_solves}'


if __name__ == '__main__':
    # test_serial_optimized_sweeper(2, 2)
    # test_RK_sweeper('IMEXEulerStifflyAccurate')
    test_parallel_sweeper(None, 2)
