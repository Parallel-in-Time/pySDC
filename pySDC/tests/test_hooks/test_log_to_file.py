import pytest


def run(hook, Tend=0):
    from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    level_params = {'dt': 1.0e-1}

    sweeper_params = {
        'num_nodes': 1,
        'quad_type': 'GAUSS',
    }

    description = {
        'level_params': level_params,
        'sweeper_class': generic_implicit,
        'problem_class': testequation0d,
        'sweeper_params': sweeper_params,
        'problem_params': {},
        'step_params': {'maxiter': 1},
    }

    controller_params = {
        'hook_class': hook,
        'logger_level': 30,
    }
    controller = controller_nonMPI(1, controller_params, description)
    if Tend > 0:
        prob = controller.MS[0].levels[0].prob
        u0 = prob.u_exact(0)

        _, stats = controller.run(u0, 0, Tend)
        return u0, stats


@pytest.mark.base
def test_errors():
    from pySDC.implementations.hooks.log_solution import LogToFile
    import os

    with pytest.raises(ValueError):
        run(LogToFile)

    LogToFile.path = os.getcwd()
    run(LogToFile)

    path = f'{os.getcwd()}/tmp'
    LogToFile.path = path
    run(LogToFile)
    os.path.isdir(path)

    with pytest.raises(ValueError):
        LogToFile.path = __file__
        run(LogToFile)


@pytest.mark.base
def test_logging():
    from pySDC.implementations.hooks.log_solution import LogToFile, LogSolution
    from pySDC.helpers.stats_helper import get_sorted
    import os
    import pickle
    import numpy as np

    path = f'{os.getcwd()}/tmp'
    LogToFile.path = path
    Tend = 2

    u0, stats = run([LogToFile, LogSolution], Tend=Tend)
    u = [(0.0, u0)] + get_sorted(stats, type='u')

    u_file = []
    for i in range(len(u)):
        data = LogToFile.load(i)
        u_file += [(data['t'], data['u'])]

    for us, uf in zip(u, u_file):
        assert us[0] == uf[0]
        assert np.allclose(us[1], uf[1])


if __name__ == '__main__':
    test_logging()
