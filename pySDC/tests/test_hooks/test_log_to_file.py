import pytest


def run(hook, Tend=0, ODE=True, t0=0):
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.helpers.fieldsIO import FieldsIO

    if ODE:
        from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d as problem_class
        from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class

        problem_params = {'u0': 1.0}
    else:
        from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard as problem_class
        from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order as sweeper_class

        problem_params = {'nx': 16, 'nz': 8, 'spectral_space': False}

    level_params = {'dt': 1.0e-2}

    sweeper_params = {
        'num_nodes': 1,
        'quad_type': 'GAUSS',
    }

    description = {
        'level_params': level_params,
        'sweeper_class': sweeper_class,
        'problem_class': problem_class,
        'sweeper_params': sweeper_params,
        'problem_params': problem_params,
        'step_params': {'maxiter': 1},
    }

    controller_params = {
        'hook_class': hook,
        'logger_level': 15,
    }
    controller = controller_nonMPI(1, controller_params, description)
    if Tend > 0:
        prob = controller.MS[0].levels[0].prob
        u0 = prob.u_exact(0)
        if t0 > 0:
            u0[:] = hook.load(-1)['u']

        _, stats = controller.run(u0, t0, Tend)
        return u0, stats


@pytest.mark.base
def test_errors_pickle():
    from pySDC.implementations.hooks.log_solution import LogToPickleFile
    import os

    with pytest.raises(ValueError):
        run(LogToPickleFile)

    LogToPickleFile.path = os.getcwd()
    run(LogToPickleFile)

    path = f'{os.getcwd()}/tmp'
    LogToPickleFile.path = path
    run(LogToPickleFile)
    os.path.isdir(path)

    with pytest.raises(ValueError):
        LogToPickleFile.path = __file__
        run(LogToPickleFile)


@pytest.mark.base
def test_errors_FieldsIO(tmpdir):
    from pySDC.implementations.hooks.log_solution import LogToFile as hook
    from pySDC.core.errors import DataError
    import os

    path = f'{tmpdir}/FieldsIO_test.pySDC'
    hook.filename = path

    run_kwargs = {'hook': hook, 'Tend': 0.2, 'ODE': True}

    # create file
    run(**run_kwargs)

    # test that we cannot overwrite if we don't want to
    hook.allow_overwriting = False
    with pytest.raises(FileExistsError):
        run(**run_kwargs)

    # test that we can overwrite if we do want to
    hook.allow_overwriting = True
    run(**run_kwargs)

    # test that we cannot add solutions at times that already exist
    hook.allow_overwriting = False
    with pytest.raises(DataError):
        run(**run_kwargs, t0=0.1)


@pytest.mark.base
@pytest.mark.parametrize('use_pickle', [True, False])
def test_logging(tmpdir, use_pickle, ODE=True):
    from pySDC.implementations.hooks.log_solution import LogToPickleFile, LogSolution, LogToFile
    from pySDC.helpers.stats_helper import get_sorted
    import os
    import pickle
    import numpy as np

    path = tmpdir
    Tend = 0.2

    if use_pickle:
        logging_hook = LogToPickleFile
        LogToPickleFile.path = path
    else:
        logging_hook = LogToFile
        logging_hook.filename = f'{path}/FieldsIO_test.pySDC'

    u0, stats = run([logging_hook, LogSolution], Tend=Tend, ODE=ODE)
    u = [(0.0, u0)] + get_sorted(stats, type='u')

    u_file = []
    for i in range(len(u)):
        data = logging_hook.load(i)
        u_file += [(data['t'], data['u'])]

    for us, uf in zip(u, u_file):
        assert us[0] == uf[0], 'time does not match'
        if ODE:
            assert np.allclose(us[1], uf[1]), 'solution does not match'
        else:
            assert np.allclose(us[1], uf[1][:4]), 'solution does not match'


@pytest.mark.base
def test_restart(tmpdir, ODE=True):
    from pySDC.implementations.hooks.log_solution import LogSolution, LogToFile
    import numpy as np

    Tend = 0.2

    # run the whole thing
    logging_hook = LogToFile
    logging_hook.filename = f'{tmpdir}/file.pySDC'

    _, _ = run([logging_hook], Tend=Tend, ODE=ODE)

    u_continuous = []
    for i in range(20):
        data = logging_hook.load(i)
        u_continuous += [(data['t'], data['u'])]

    # run again with a restart in the middle
    logging_hook.filename = f'{tmpdir}/file2.pySDC'
    _, _ = run(logging_hook, Tend=0.1, ODE=ODE)
    _, _ = run(logging_hook, Tend=0.2, t0=0.1, ODE=ODE)

    u_restart = []
    for i in range(20):
        data = logging_hook.load(i)
        u_restart += [(data['t'], data['u'])]

    assert np.allclose([me[0] for me in u_restart], [me[0] for me in u_continuous]), 'Times don\'t match'
    for u1, u2 in zip(u_restart, u_continuous):
        assert np.allclose(u1[1], u2[1]), 'solution does not match'


@pytest.mark.mpi4py
@pytest.mark.mpi(ranks=[1, 4])
def test_loggingMPI(tmpdir, comm, mpi_ranks):
    # `mpi_ranks` is a pytest fixture required by pytest-isolate-mpi. Do not remove.
    tmpdir = comm.bcast(tmpdir)
    test_logging(tmpdir, False, False)
