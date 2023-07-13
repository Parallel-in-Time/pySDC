import pytest

SWEEPER_NAMES = [
    'ForwardEuler',
    'ExplicitMidpointMethod',
    'CrankNicholson',
    'BackwardEuler',
    'ImplicitMidpointMethod',
    'RK4',
    'Cash_Karp',
    'ESDIRK53',
    'DIRK43',
]


def get_sweeper(sweeper_name):
    """
    Retrieve a sweeper from a name

    Args:
        sweeper_name (str):

    Returns:
        pySDC.Sweeper.RungeKutta: The sweeper
    """
    import pySDC.implementations.sweeper_classes.Runge_Kutta as RK

    return eval(f'RK.{sweeper_name}')


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
        pySDC.Controller.controller: Controller
    """
    from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.hooks.log_work import LogWork
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun
    from pySDC.implementations.hooks.log_solution import LogSolution
    from pySDC.implementations.hooks.log_embedded_error_estimate import LogEmbeddedErrorEstimate
    from pySDC.implementations.convergence_controller_classes.estimate_embedded_error import EstimateEmbeddedError

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
        'convergence_controllers': {EstimateEmbeddedError: {}},
    }

    controller_params = {
        'logger_level': 30,
        'hook_class': [LogWork, LogGlobalErrorPostRun, LogSolution, LogEmbeddedErrorEstimate],
    }

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    prob = controller.MS[0].levels[0].prob
    ic = prob.u_exact(0)
    u_end, stats = controller.run(ic, 0.0, Tend)

    return stats, ic, controller


@pytest.mark.base
@pytest.mark.parametrize("sweeper_name", SWEEPER_NAMES)
def test_order(sweeper_name):
    """
    Test the order in time of the method

    Args:
        sweeper_name (str): Name of the RK method
    """
    import numpy as np

    from pySDC.helpers.stats_helper import get_sorted

    expected_order = {
        'ForwardEuler': 1,
        'BackwardEuler': 1,
        'ExplicitMidpointMethod': 2,
        'ImplicitMidpointMethod': 2,
        'RK4': 4,
        'CrankNicholson': 2,
        'Cash_Karp': 5,
        'ESDIRK53': 5,
        'DIRK43': 4,
    }

    expected_embedded_order = {
        'DIRK43': 4,
        'Cash_Karp': 5,
        'ESDIRK53': 4,
    }

    dt_max = {
        'Cash_Karp': 1e0,
        'ESDIRK53': 1e0,
    }

    lambdas = [[-1.0e-1 + 0j]]

    e = {}
    e_embedded = {}
    dts = [dt_max.get(sweeper_name, 1e-1) / 2**i for i in range(5)]
    for dt in dts:
        stats, _, _ = single_run(sweeper_name, dt, 2 * max(dts), lambdas)
        e[dt] = get_sorted(stats, type='e_global_post_run')[-1][1]
        e_em = get_sorted(stats, type='error_embedded_estimate')
        if len(e_em):
            e_embedded[dt] = e_em[-1][1]
        else:
            e_embedded[dt] = 0.0

    order = [
        np.log(e[dts[i]] / e[dts[i + 1]]) / np.log(dts[i] / dts[i + 1])
        for i in range(len(dts) - 1)
        if e[dts[i + 1]] > 1e-14
    ]
    order_embedded = [
        np.log(e_embedded[dts[i]] / e_embedded[dts[i + 1]]) / np.log(dts[i] / dts[i + 1])
        for i in range(len(dts) - 1)
        if e_embedded[dts[i + 1]] > 1e-14
    ]

    assert np.isclose(
        np.mean(order), expected_order[sweeper_name], atol=0.2
    ), f"Got unexpected order {np.mean(order):.2f} for {sweeper_name} method! ({order})"

    if sweeper_name in expected_embedded_order.keys():
        assert np.isclose(
            np.mean(order_embedded), expected_embedded_order[sweeper_name], atol=0.2
        ), f"Got unexpected order of embedded error estimate {np.mean(order_embedded):.2f} for {sweeper_name} method! ({order_embedded})"


@pytest.mark.base
@pytest.mark.parametrize("sweeper_name", SWEEPER_NAMES)
def test_stability(sweeper_name):
    """
    Test the stability of the method

    Args:
        sweeper_name (str): Name of the RK method
    """
    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted

    expected_A_stability = {
        'ForwardEuler': False,
        'BackwardEuler': True,
        'ExplicitMidpointMethod': False,
        'ImplicitMidpointMethod': True,
        'RK4': False,
        'CrankNicholson': True,
        'Cash_Karp': False,
        'ESDIRK53': True,
        'DIRK43': True,
    }

    re = -np.logspace(-3, 6, 50)
    im = -np.logspace(-3, 6, 50)
    lambdas = np.array([[complex(re[i], im[j]) for i in range(len(re))] for j in range(len(im))]).reshape(
        (len(re) * len(im))
    )

    stats, ic, _ = single_run(sweeper_name, 1.0, 1.0, lambdas)
    u = get_sorted(stats, type='u')[-1][1]

    unstable = np.abs(u[np.abs(ic) > 0]) / np.abs(ic[np.abs(ic) > 0]) > 1.0

    Astable = not any(lambdas[unstable].real < 0)
    assert Astable == expected_A_stability[sweeper_name], f"Unexpected stability properties for {sweeper_name} method!"
    assert any(~unstable), f"{sweeper_name} method is stable nowhere!"


@pytest.mark.base
@pytest.mark.parametrize("sweeper_name", SWEEPER_NAMES)
def test_rhs_evals(sweeper_name):
    """
    Test the number of right hand side evaluations.

    Args:
        sweeper_name (str): Name of the RK method
    """
    from pySDC.helpers.stats_helper import get_sorted

    lambdas = [[-1.0e-1 + 0j]]

    stats, _, controller = single_run(sweeper_name, 1.0, 10.0, lambdas)

    sweep = controller.MS[0].levels[0].sweep
    num_stages = sweep.coll.num_nodes - sweep.coll.num_solution_stages

    rhs_evaluations = [me[1] for me in get_sorted(stats, type='work_rhs')]

    assert all(
        me == num_stages for me in rhs_evaluations
    ), f'Did not perform one RHS evaluation per step and stage in {sweeper_name} method! Expected {num_stages}, but got {rhs_evaluations}.'


@pytest.mark.base
def test_embedded_method():
    """
    Here, we test if Cash Karp's method gives a hard-coded result and number of restarts when running with adaptivity.

    Returns:
        None
    """
    import numpy as np
    from pySDC.projects.Resilience.vdp import run_vdp
    from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityRK
    from pySDC.helpers.stats_helper import get_sorted

    sweeper_name = 'Cash_Karp'

    # change only the things in the description that we need for adaptivity
    adaptivity_params = {}
    adaptivity_params['e_tol'] = 1e-7

    convergence_controllers = {}
    convergence_controllers[AdaptivityRK] = adaptivity_params

    description = {}
    description['convergence_controllers'] = convergence_controllers
    description['sweeper_class'] = get_sweeper(sweeper_name)
    description['step_params'] = {'maxiter': 1}

    custom_controller_params = {'logger_level': 40}

    stats, _, _ = run_vdp(description, 1, custom_controller_params=custom_controller_params)

    dt_last = get_sorted(stats, type='dt')[-2][1]
    restarts = sum([me[1] for me in get_sorted(stats, type='restart')])
    assert np.isclose(dt_last, 0.14175080252629996), "Cash-Karp has computed a different last step size than before!"
    assert restarts == 17, "Cash-Karp has restarted a different number of times than before"


if __name__ == '__main__':
    test_order('ESDIRK53')
