import pytest


def run(hook, Tend=0, ODE=True, t0=0, dt=1.0e-2):
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

    level_params = {'dt': dt}

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
def test_errors_pickle(tmp_path):
    from pySDC.implementations.hooks.log_solution import LogToPickleFile
    import os

    hook = type('LogToPickleFile', (LogToPickleFile,), {})

    with pytest.raises(ValueError, match='Please set a path'):
        run(hook)

    hook.path = str(tmp_path)
    run(hook)

    # a directory that does not exist yet is created
    path = f'{tmp_path}/tmp'
    hook.path = path
    run(hook)
    assert os.path.isdir(path)

    hook.path = __file__
    with pytest.raises(ValueError, match='a file of the same name exists'):
        run(hook)


@pytest.mark.base
@pytest.mark.parametrize('hook_name', ['LogToPickleFile', 'LogToPickleFileAfterXS', 'LogToFile'])
def test_hooks_pass_calls_on_to_the_next_class(tmp_path, hook_name):
    """Combined with another hook by inheritance, the other hook still records its stats."""
    import pySDC.implementations.hooks.log_solution as log_solution
    from pySDC.implementations.hooks.log_solution import LogSolution
    from pySDC.helpers.stats_helper import get_sorted

    attrs = {'path': str(tmp_path), 'filename': f'{tmp_path}/combined.pySDC', 'time_increment': 1.0, 'counter': 0}
    hook = type('Combined', (getattr(log_solution, hook_name), LogSolution), attrs)

    _, stats = run(hook, Tend=0.03)
    assert len(get_sorted(stats, type='u')) == 3


@pytest.mark.base
def test_plots_are_labelled_with_their_time():
    """The initial conditions are plotted at the start of the run and the solution at the end of each step."""
    import numpy as np
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.hooks.plotting import PlotPostStep
    from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit

    times = []

    class RecordingProblem(testequation0d):
        def get_fig(self):
            return None

        def plot(self, u, t=None, fig=None):
            times.append(t)

    hook = type('Plot', (PlotPostStep,), {'live_plot': None})
    description = {
        'problem_class': RecordingProblem,
        'problem_params': {},
        'sweeper_class': generic_implicit,
        'sweeper_params': {'num_nodes': 1, 'quad_type': 'GAUSS'},
        'level_params': {'dt': 0.1},
        'step_params': {'maxiter': 1},
    }
    controller = controller_nonMPI(1, {'hook_class': hook, 'logger_level': 30}, description)
    prob = controller.MS[0].levels[0].prob
    controller.run(prob.u_exact(0), t0=1.0, Tend=1.3)

    assert np.allclose(times, [1.0, 1.1, 1.2, 1.3]), times


@pytest.mark.base
def test_errors_FieldsIO(tmpdir):
    from pySDC.implementations.hooks.log_solution import LogToFile
    from pySDC.core.errors import DataError
    import os

    path = f'{tmpdir}/FieldsIO_test.pySDC'

    class hook(LogToFile):
        filename = path

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
@pytest.mark.parametrize('hook_name', ['LogToPickleFileAfterXS', 'LogToFile'])
def test_logging_times(tmpdir, hook_name):
    """Steps finer than the time increment log at its multiples, although the accumulated times round away from them."""
    import pySDC.implementations.hooks.log_solution as log_solution
    import numpy as np

    attrs = {'path': str(tmpdir), 'filename': f'{tmpdir}/times.pySDC', 'time_increment': 0.1, 'counter': 0}
    hook = type(hook_name, (getattr(log_solution, hook_name),), attrs)
    run(hook, Tend=0.5, dt=0.05)

    times = [hook.load(i)['t'] for i in range(hook.counter)]
    assert np.allclose(times, np.arange(6) * 0.1), times


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
        logging_hook = type('LogToPickleFile', (LogToPickleFile,), {'path': path})
    else:
        logging_hook = type('LogToFile', (LogToFile,), {'filename': f'{path}/FieldsIO_test.pySDC'})

    u0, stats = run([logging_hook, LogSolution], Tend=Tend, ODE=ODE)
    u = [(0.0, u0)] + get_sorted(stats, type='u')

    u_file = []
    for i in range(len(u)):
        data = logging_hook.load(i)
        u_file += [(data['t'], data['u'])]

    for us, uf in zip(u, u_file, strict=True):
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
    logging_hook = type('LogToFile', (LogToFile,), {'filename': f'{tmpdir}/file.pySDC'})

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
    for u1, u2 in zip(u_restart, u_continuous, strict=True):
        assert np.allclose(u1[1], u2[1]), 'solution does not match'


@pytest.mark.mpi4py
@pytest.mark.parallel([1, 4])
def test_loggingMPI():
    import tempfile
    import shutil
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    tmpdir = tempfile.mkdtemp() if comm.rank == 0 else None
    tmpdir = comm.bcast(tmpdir, root=0)
    try:
        test_logging(tmpdir, False, False)
    finally:
        if comm.rank == 0:
            shutil.rmtree(tmpdir, ignore_errors=True)
