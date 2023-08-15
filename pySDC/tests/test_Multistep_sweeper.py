import pytest

sweeper_names = [
    'AdamsMoultonImplicit2Step',
    'AdamsMoultonImplicit1Step',
    'BackwardEuler',
    'AdamsBashforthExplicit1Step',
]


def get_sweeper(sweeper_name):
    """
    Retrieve a sweeper from a name

    Args:
        sweeper_name (str):

    Returns:
        pySDC.Sweeper.RungeKutta: The sweeper
    """
    import pySDC.implementations.sweeper_classes.Multistep as Multistep

    return eval(f'Multistep.{sweeper_name}')


def single_run(sweeper_name, dt, Tend, lambdas):
    """
    Do a single run of the test equation.

    Args:
        sweeper_name (str): Name of Multistep method
        dt (float): Step size to use
        Tend (float): Time to simulate to
        lambdas (2d complex numpy.ndarray): Lambdas for test equation

    Returns:
        dict: Stats
        pySDC.datatypes.mesh: Initial conditions
    """
    from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.hooks.log_work import LogWork
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun
    from pySDC.implementations.hooks.log_solution import LogSolution

    level_params = {'dt': dt}

    step_params = {'maxiter': 1}

    problem_params = {
        'lambdas': lambdas,
        'u0': 1.0 + 0.0j,
    }

    sweeper_params = {
        'num_nodes': 1,
        'quad_type': 'RADAU-RIGHT',
    }

    description = {
        'level_params': level_params,
        'step_params': step_params,
        'sweeper_class': get_sweeper(sweeper_name),
        'problem_class': testequation0d,
        'sweeper_params': sweeper_params,
        'problem_params': problem_params,
    }

    controller_params = {
        'logger_level': 30,
        'hook_class': [LogWork, LogGlobalErrorPostRun, LogSolution],
    }

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    prob = controller.MS[0].levels[0].prob
    ic = prob.u_exact(0)
    u_end, stats = controller.run(ic, 0.0, Tend)

    return stats, ic


@pytest.mark.base
@pytest.mark.parametrize("sweeper_name", sweeper_names)
def test_order(sweeper_name):
    """
    Test the order in time of the method

    Args:
        sweeper_name (str): Name of the multistep method
    """
    import numpy as np

    from pySDC.helpers.stats_helper import get_sorted

    expected_order = {
        'AdamsMoultonImplicit1Step': 2,
        'AdamsMoultonImplicit2Step': 3,
        'BackwardEuler': 1,
        'AdamsBashforthExplicit1Step': 1,
    }

    dt_max = {}

    lambdas = [[-1.0e-1 + 0j]]

    e = {}
    dts = [dt_max.get(sweeper_name, 1e-1) / 2**i for i in range(5)]
    for dt in dts:
        stats, _ = single_run(sweeper_name, dt, 2 * max(dts), lambdas)
        e[dt] = get_sorted(stats, type='e_global_post_run')[-1][1]

    order = [np.log(e[dts[i]] / e[dts[i + 1]]) / np.log(dts[i] / dts[i + 1]) for i in range(len(dts) - 1)]

    assert np.isclose(
        np.mean(order), expected_order[sweeper_name], atol=0.2
    ), f"Got unexpected order {np.mean(order):.2f} for {sweeper_name} method!"


@pytest.mark.base
@pytest.mark.parametrize("sweeper_name", sweeper_names)
def test_stability(sweeper_name):
    """
    Test the stability of the method

    Args:
        sweeper_name (str): Name of the multistep method
    """
    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted

    expected_A_stability = {
        'AdamsMoultonImplicit1Step': True,
        'AdamsMoultonImplicit2Step': True,
        'BackwardEuler': True,
        'AdamsBashforthExplicit1Step': False,
    }

    re = -np.logspace(-3, 2, 50)
    im = -np.logspace(-3, 2, 50)
    lambdas = np.array([[complex(re[i], im[j]) for i in range(len(re))] for j in range(len(im))]).reshape(
        (len(re) * len(im))
    )

    stats, ic = single_run(sweeper_name, 1.0, 1.0, lambdas)
    u = get_sorted(stats, type='u')[-1][1]
    unstable = np.abs(u) / np.abs(ic) > 1.0

    Astable = not any(lambdas[unstable].real < 0)
    assert Astable == expected_A_stability[sweeper_name], f"Unexpected stability properties for {sweeper_name} method!"
    assert any(~unstable), f"{sweeper_name} method is stable nowhere!"


@pytest.mark.base
@pytest.mark.parametrize("sweeper_name", sweeper_names)
def test_rhs_evals(sweeper_name):
    """
    Test the number of right hand side evaluations.

    Args:
        sweeper_name (str): Name of the multistep method
    """
    from pySDC.helpers.stats_helper import get_sorted

    # record how many right hand side evaluations we expect in the format ([<rhs evals during startup phase>], <rhs side evals per step after startup phase>)
    expected = {
        'AdamsMoultonImplicit1Step': ([2], 1),
        'AdamsMoultonImplicit2Step': ([2], 1),
        'BackwardEuler': ([2], 1),
        'AdamsBashforthExplicit1Step': ([2], 1),
    }

    lambdas = [[-1.0e-1 + 0j]]

    stats, _ = single_run(sweeper_name, 1.0, 10.0, lambdas)

    rhs_evals = [me[1] for me in get_sorted(stats, type='work_rhs')]

    startup_phase_expected = expected[sweeper_name][0]
    len_startup = len(startup_phase_expected)
    assert all(
        rhs_evals[i] == startup_phase_expected[i] for i in range(len_startup)
    ), f'Unexpected number of rhs evaluations during startup phase for {sweeper_name} method!'

    assert all(
        rhs_evals[i] == expected[sweeper_name][1] for i in range(len_startup, len(rhs_evals))
    ), f'Unexpected number of rhs evaluations after startup phase for {sweeper_name} method!'


if __name__ == '__main__':
    test_rhs_evals('AdamsMoultonImplicit2Step')
