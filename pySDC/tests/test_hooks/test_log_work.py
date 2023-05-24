import pytest

# @pytest.mark.
def run_Lorenz(useMPI, maxiter=4, newton_maxiter=5, num_procs=1):
    from pySDC.implementations.hooks.log_work import LogWork
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.problem_classes.Lorenz import LorenzAttractor
    from pySDC.helpers.stats_helper import get_sorted

    num_steps = 2

    # initialize level parameters
    level_params = {}
    level_params['dt'] = 1e-2
    level_params['restol'] = -1

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 1
    sweeper_params['QI'] = 'IE'

    problem_params = {
        'newton_tol': -1,  # force to iterate to `newton_maxiter`
        'newton_maxiter': newton_maxiter,
    }

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = maxiter

    # initialize controller parameters
    controller_params = {}
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = LogWork
    controller_params['mssdc_jac'] = False

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = LorenzAttractor
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # set time parameters
    t0 = 0.0

    # instantiate controller
    if useMPI:
        from mpi4py import MPI
        from pySDC.implementations.controller_classes.controller_MPI import controller_MPI

        comm = MPI.COMM_WORLD
        num_procs = comm.size

        controller = controller_MPI(controller_params=controller_params, description=description, comm=comm)
        P = controller.S.levels[0].prob
    else:
        from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

        comm = None
        controller = controller_nonMPI(
            num_procs=num_procs, controller_params=controller_params, description=description
        )
        P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    uend, stats = controller.run(u0=uinit, t0=t0, Tend=num_steps * num_procs * level_params['dt'])

    for i in range(num_procs):
        res = {
            key: [me[1] for me in get_sorted(stats, type=key, comm=comm, process=i)]
            for key in ['work_newton', 'work_rhs']
        }

        expected = {}
        if i == 0:
            # we evaluate all nodes when beginning the step and then every node except the initial conditions in every iteration
            expected['work_rhs'] = maxiter * sweeper_params['num_nodes'] + sweeper_params['num_nodes'] + 1
        else:
            # Additionally, we reevaluate what we received. Once before we start iterating and then whenever we start a new iteration and in `it_check`
            expected['work_rhs'] = maxiter * (sweeper_params['num_nodes'] + 2) + sweeper_params['num_nodes'] + 2

        expected['work_newton'] = newton_maxiter * sweeper_params['num_nodes'] * maxiter

        for key, val in res.items():
            assert all(
                me == expected[key] for me in val
            ), f'Error in LogWork hook when recording \"{key}\" for process {i}! Got {val}, expected {expected[key]}!'

    return None


@pytest.mark.mpi4py
@pytest.mark.parametrize("num_procs", [1, 3])
@pytest.mark.parametrize("maxiter", [0, 3])
@pytest.mark.parametrize("newton_maxiter", [1, 3])
def test_LogWork_MPI(num_procs, newton_maxiter, maxiter):
    import os
    import subprocess

    kwargs = {}
    kwargs['useMPI'] = 1
    kwargs['num_procs'] = num_procs
    kwargs['newton_maxiter'] = newton_maxiter
    kwargs['maxiter'] = maxiter

    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

    # run code with different number of MPI processes
    kwargs_str = "".join([f"{key}:{item} " for key, item in kwargs.items()])
    cmd = f"mpirun -np {num_procs} python {__file__} {kwargs_str}".split()

    p = subprocess.Popen(cmd, env=my_env, cwd=".")

    p.wait()
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (
        p.returncode,
        num_procs,
    )


@pytest.mark.base
@pytest.mark.parametrize("num_procs", [1, 3])
@pytest.mark.parametrize("maxiter", [0, 3])
@pytest.mark.parametrize("newton_maxiter", [1, 3])
def test_LogWork_nonMPI(num_procs, newton_maxiter, maxiter):
    kwargs = {}
    kwargs['useMPI'] = 0
    kwargs['num_procs'] = num_procs
    kwargs['newton_maxiter'] = newton_maxiter
    kwargs['maxiter'] = maxiter
    run_Lorenz(**kwargs)


if __name__ == "__main__":
    import sys

    kwargs = {me.split(':')[0]: int(me.split(':')[1]) for me in sys.argv[1:]}
    run_Lorenz(**kwargs)
